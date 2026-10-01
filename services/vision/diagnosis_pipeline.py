"""End-to-end orchestration for grounded multimodal diagnosis."""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Optional

import httpx

from services.vision.image_preprocessor import (
    ImagePreprocessingResult,
    preprocess_image,
)
from services.vision.ocr_engine import PaddleOCREngine
from services.vision.result_fusion import fuse_multimodal_results
from services.vision.schemas import (
    MultimodalFusionResult,
    OCRExtraction,
    VisionInferenceResult,
)
from services.vision.vision_client import OllamaVisionClient


DEFAULT_ENBOT_API_URL = "http://127.0.0.1:8000"
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")


class DiagnosisPipelineError(RuntimeError):
    """Base error raised by the multimodal diagnosis pipeline."""


class EnBotAPIUnavailableError(DiagnosisPipelineError):
    """Raised when the existing text/RAG API cannot be reached."""


class EnBotAPIResponseError(DiagnosisPipelineError):
    """Raised when the existing text/RAG API returns an invalid response."""


@dataclass(frozen=True)
class MultimodalDiagnosisResult:
    """Completed diagnosis with evidence, answer, and stage timings."""

    status: str
    answer: str
    request_id: Optional[str]
    sources: tuple[str, ...]
    retrieval_backend: Optional[str]
    preprocessing: ImagePreprocessingResult
    ocr: OCRExtraction
    vision: VisionInferenceResult
    fusion: MultimodalFusionResult
    total_seconds: float
    stage_seconds: dict[str, float]

    def to_dict(self, *, include_debug: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "status": self.status,
            "answer": self.answer,
            "request_id": self.request_id,
            "sources": list(self.sources),
            "retrieval_backend": self.retrieval_backend,
            "fusion": self.fusion.to_dict(),
            "total_seconds": self.total_seconds,
            "stage_seconds": dict(self.stage_seconds),
        }
        if include_debug:
            data["preprocessing"] = self.preprocessing.to_dict()
            data["ocr"] = asdict(self.ocr)
            data["vision"] = self.vision.to_dict()
        return data


def _env_positive_float(name: str, default: float) -> float:
    raw_value = os.getenv(name, "").strip()
    value = float(raw_value) if raw_value else default
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return value


def _timed_call(function: Callable[..., Any], *args: Any) -> tuple[Any, float]:
    started = perf_counter()
    value = function(*args)
    return value, perf_counter() - started


class MultimodalDiagnosisPipeline:
    """Compose image processing, local extraction, fusion, and EnBot RAG."""

    def __init__(
        self,
        *,
        preprocessor: Callable[..., ImagePreprocessingResult] = preprocess_image,
        ocr_engine: Optional[PaddleOCREngine] = None,
        vision_client: Optional[OllamaVisionClient] = None,
        enbot_api_url: Optional[str] = None,
        enbot_api_key: Optional[str] = None,
        enbot_timeout_seconds: Optional[float] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self._preprocessor = preprocessor
        self._ocr_engine = ocr_engine or PaddleOCREngine()
        self._vision_client = vision_client or OllamaVisionClient()
        self._owns_vision_client = vision_client is None
        self.enbot_api_url = (
            enbot_api_url
            or os.getenv("ENBOT_API_BASE_URL", DEFAULT_ENBOT_API_URL)
        ).rstrip("/")
        self.enbot_api_key = (
            enbot_api_key
            if enbot_api_key is not None
            else os.getenv("ENBOT_API_KEY", "").strip()
        )
        self.enbot_timeout_seconds = (
            enbot_timeout_seconds
            if enbot_timeout_seconds is not None
            else _env_positive_float("ENBOT_API_TIMEOUT_SECONDS", 300.0)
        )
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.Client(
            base_url=self.enbot_api_url,
            timeout=httpx.Timeout(self.enbot_timeout_seconds),
        )

    def close(self) -> None:
        if self._owns_http_client:
            self._http_client.close()
        if self._owns_vision_client:
            self._vision_client.close()

    def __enter__(self) -> "MultimodalDiagnosisPipeline":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    def _call_enbot(self, query: str, session_id: str) -> Mapping[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.enbot_api_key:
            headers["X-API-Key"] = self.enbot_api_key
        payload = {
            "query": query,
            "session_id": session_id,
            "evaluate": False,
            "debug": False,
        }
        try:
            response = self._http_client.post(
                "/v1/chat",
                headers=headers,
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise EnBotAPIUnavailableError(
                f"EnBot API timed out after {self.enbot_timeout_seconds:g} seconds"
            ) from exc
        except httpx.RequestError as exc:
            raise EnBotAPIUnavailableError(
                f"Cannot connect to EnBot API at {self.enbot_api_url}: {exc}"
            ) from exc

        if response.status_code >= 400:
            preview = response.text.strip().replace("\n", " ")[:500]
            raise EnBotAPIResponseError(
                f"EnBot API returned HTTP {response.status_code}: {preview}"
            )
        try:
            response_payload = response.json()
        except (ValueError, TypeError) as exc:
            raise EnBotAPIResponseError("EnBot API returned invalid JSON") from exc
        if not isinstance(response_payload, Mapping):
            raise EnBotAPIResponseError("EnBot API response must be a JSON object")
        return response_payload

    def run(
        self,
        image_path: str | Path,
        *,
        question: Optional[str] = None,
        session_id: str = "vision-default",
    ) -> MultimodalDiagnosisResult:
        """Run the complete pipeline and call RAG only for in-domain evidence."""

        if not _SESSION_ID_PATTERN.fullmatch(session_id):
            raise ValueError(
                "session_id must be 1-128 characters using A-Z, a-z, 0-9, _.:@-"
            )
        normalized_question = (question or "").strip()
        if len(normalized_question) > 4000:
            raise ValueError("question must not exceed 4000 characters")
        if not normalized_question:
            normalized_question = "Please analyze the device status or issue shown in this image."

        pipeline_started = perf_counter()
        preprocessing_started = perf_counter()
        preprocessing = self._preprocessor(image_path)
        preprocessing_seconds = perf_counter() - preprocessing_started

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="vision-extract") as pool:
            ocr_future = pool.submit(
                _timed_call,
                self._ocr_engine.extract,
                preprocessing.ocr_image_path,
            )
            vision_future = pool.submit(
                _timed_call,
                self._vision_client.analyze,
                preprocessing.vlm_image_path,
            )
            ocr, ocr_seconds = ocr_future.result()
            vision, vision_seconds = vision_future.result()

        fusion_started = perf_counter()
        fusion = fuse_multimodal_results(
            ocr,
            vision,
            user_question=normalized_question,
        )
        fusion_seconds = perf_counter() - fusion_started

        rag_seconds = 0.0
        request_id: Optional[str] = None
        sources: tuple[str, ...] = ()
        retrieval_backend: Optional[str] = None

        if not fusion.retrieval_allowed:
            manufacturer = fusion.manufacturer or "an unverified or external manufacturer"
            answer = (
                "The image is not verified as an EnValt/Hanchu product "
                f"(detected manufacturer: {manufacturer}). To avoid an incorrect "
                "diagnosis, the EnBot knowledge base was not queried."
            )
            status = "out_of_domain"
        else:
            rag_started = perf_counter()
            enbot_payload = self._call_enbot(fusion.retrieval_query, session_id)
            rag_seconds = perf_counter() - rag_started
            data = enbot_payload.get("data")
            if not isinstance(data, Mapping):
                raise EnBotAPIResponseError("EnBot API response is missing data")
            answer_value = data.get("response")
            if not isinstance(answer_value, str) or not answer_value.strip():
                raise EnBotAPIResponseError("EnBot API response is missing answer text")
            source_values = data.get("sources") or []
            if not isinstance(source_values, list):
                raise EnBotAPIResponseError("EnBot API sources must be a list")
            answer = answer_value.strip()
            request_id_value = enbot_payload.get("request_id")
            request_id = str(request_id_value) if request_id_value else None
            sources = tuple(str(item) for item in source_values if item)
            backend_value = data.get("retrieval_backend")
            retrieval_backend = str(backend_value) if backend_value else None
            status = "success"

        total_seconds = perf_counter() - pipeline_started
        return MultimodalDiagnosisResult(
            status=status,
            answer=answer,
            request_id=request_id,
            sources=sources,
            retrieval_backend=retrieval_backend,
            preprocessing=preprocessing,
            ocr=ocr,
            vision=vision,
            fusion=fusion,
            total_seconds=round(total_seconds, 6),
            stage_seconds={
                "preprocessing": round(preprocessing_seconds, 6),
                "ocr": round(ocr_seconds, 6),
                "vision": round(vision_seconds, 6),
                "fusion": round(fusion_seconds, 6),
                "rag": round(rag_seconds, 6),
            },
        )
