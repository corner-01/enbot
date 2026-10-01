"""Strict local Ollama client for structured visual evidence extraction."""

from __future__ import annotations

import base64
import json
import os
import re
import threading
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Optional

import httpx
from pydantic import ValidationError

from services.vision.prompts import (
    VISUAL_EXTRACTION_PROMPT_VERSION,
    build_visual_extraction_prompt,
)
from services.vision.schemas import VisualExtraction, VisionInferenceResult


DEFAULT_MODEL = "qwen3-vl:2b-instruct"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"


class VisionClientError(RuntimeError):
    """Base error for local visual inference failures."""


class VisionServiceUnavailableError(VisionClientError):
    """Raised when Ollama cannot be reached or times out."""


class VisionModelResponseError(VisionClientError):
    """Raised when Ollama or the model returns an invalid response."""


def _env_positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    value = int(raw_value) if raw_value else default
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _env_positive_float(name: str, default: float) -> float:
    raw_value = os.getenv(name, "").strip()
    value = float(raw_value) if raw_value else default
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return value


def _nanoseconds_to_seconds(value: Any) -> float:
    try:
        return float(value or 0) / 1_000_000_000
    except (TypeError, ValueError):
        return 0.0


_DOMAIN_BRAND_MARKERS = ("envalt", "hanchu")
_FAULT_TERMS = (
    "fault",
    "error",
    "alarm",
    "warning",
    "failure",
    "故障",
    "错误",
    "告警",
    "报警",
    "异常",
)
_FAULT_CODE_PATTERN = re.compile(r"\b(?:e|f)\d{2,}\b", flags=re.IGNORECASE)


def _normalize_visual_extraction(
    extraction: VisualExtraction,
) -> tuple[VisualExtraction, tuple[str, ...]]:
    """Apply conservative business guards without inventing visual evidence."""

    updates: dict[str, Any] = {}
    warnings: list[str] = []
    manufacturer = (extraction.manufacturer or "").strip()
    if manufacturer and not any(
        marker in manufacturer.casefold() for marker in _DOMAIN_BRAND_MARKERS
    ):
        if not extraction.out_of_domain:
            warnings.append(
                "out_of_domain forced to true because the visible manufacturer "
                f"is not EnValt/Hanchu: {manufacturer}"
            )
        updates["out_of_domain"] = True

    explicit_faults = [
        item
        for item in extraction.possible_faults
        if any(term in item.casefold() for term in _FAULT_TERMS)
        or _FAULT_CODE_PATTERN.search(item)
    ]
    if len(explicit_faults) != len(extraction.possible_faults):
        warnings.append(
            "Removed possible_faults entries that had no explicit fault, alarm, "
            "warning, or fault-code evidence."
        )
        updates["possible_faults"] = explicit_faults

    normalized = extraction.model_copy(update=updates) if updates else extraction
    return normalized, tuple(warnings)


class OllamaVisionClient:
    """Call one local vision model at a time and validate every response."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_concurrency: Optional[int] = None,
        num_predict: Optional[int] = None,
        keep_alive: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.model = model or os.getenv("VISION_MODEL", DEFAULT_MODEL).strip()
        if not self.model:
            raise ValueError("VISION_MODEL cannot be empty")
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else _env_positive_float("VISION_REQUEST_TIMEOUT_SECONDS", 120.0)
        )
        concurrency = (
            max_concurrency
            if max_concurrency is not None
            else _env_positive_int("VISION_MAX_CONCURRENCY", 1)
        )
        if concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self.num_predict = (
            num_predict
            if num_predict is not None
            else _env_positive_int("VISION_NUM_PREDICT", 1600)
        )
        if self.num_predict < 1:
            raise ValueError("num_predict must be at least 1")
        self.keep_alive = keep_alive or os.getenv("VISION_KEEP_ALIVE", "5m").strip()
        self._semaphore = threading.BoundedSemaphore(concurrency)
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout_seconds),
        )

    def close(self) -> None:
        if self._owns_http_client:
            self._http_client.close()

    def __enter__(self) -> "OllamaVisionClient":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    def analyze(self, image_path: str | Path) -> VisionInferenceResult:
        """Extract structured evidence from one preprocessed RGB image."""

        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Vision image does not exist: {path}")
        image_base64 = base64.b64encode(path.read_bytes()).decode("ascii")
        schema = VisualExtraction.model_json_schema()
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": build_visual_extraction_prompt(schema),
                    "images": [image_base64],
                }
            ],
            "format": schema,
            "stream": False,
            "options": {
                "temperature": 0,
                "seed": 42,
                "num_predict": self.num_predict,
            },
            "keep_alive": self.keep_alive,
        }

        started = perf_counter()
        try:
            with self._semaphore:
                response = self._http_client.post("/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise VisionServiceUnavailableError(
                f"Ollama vision request timed out after {self.timeout_seconds:g} seconds"
            ) from exc
        except httpx.RequestError as exc:
            raise VisionServiceUnavailableError(
                f"Cannot connect to Ollama at {self.base_url}: {exc}"
            ) from exc
        wall_seconds = perf_counter() - started

        if response.status_code >= 400:
            response_preview = response.text.strip().replace("\n", " ")[:500]
            raise VisionClientError(
                f"Ollama returned HTTP {response.status_code}: {response_preview}"
            )

        try:
            response_payload = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise VisionModelResponseError("Ollama returned invalid JSON") from exc
        if not isinstance(response_payload, Mapping):
            raise VisionModelResponseError("Ollama response must be a JSON object")

        message = response_payload.get("message")
        if not isinstance(message, Mapping):
            raise VisionModelResponseError("Ollama response is missing message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise VisionModelResponseError("Vision model returned empty content")

        try:
            extraction = VisualExtraction.model_validate_json(content)
        except (ValidationError, ValueError) as exc:
            preview = content.strip().replace("\n", " ")[:500]
            if response_payload.get("done_reason") == "length":
                raise VisionModelResponseError(
                    "Vision model output was truncated by the generation limit; "
                    f"VISION_NUM_PREDICT={self.num_predict}"
                ) from exc
            raise VisionModelResponseError(
                f"Vision model output failed schema validation: {preview}"
            ) from exc
        extraction, normalization_warnings = _normalize_visual_extraction(extraction)

        total_seconds = _nanoseconds_to_seconds(response_payload.get("total_duration"))
        load_seconds = _nanoseconds_to_seconds(response_payload.get("load_duration"))
        eval_seconds = _nanoseconds_to_seconds(response_payload.get("eval_duration"))
        prompt_tokens = int(response_payload.get("prompt_eval_count") or 0)
        generated_tokens = int(response_payload.get("eval_count") or 0)
        tokens_per_second = generated_tokens / eval_seconds if eval_seconds > 0 else 0.0

        return VisionInferenceResult(
            model=str(response_payload.get("model") or self.model),
            prompt_version=VISUAL_EXTRACTION_PROMPT_VERSION,
            image_path=str(path),
            extraction=extraction,
            normalization_warnings=normalization_warnings,
            wall_seconds=round(wall_seconds, 6),
            total_seconds=round(total_seconds, 6),
            load_seconds=round(load_seconds, 6),
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            tokens_per_second=round(tokens_per_second, 6),
        )
