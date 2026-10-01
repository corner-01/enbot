"""Lazy, CPU-only PaddleOCR adapter for exact technical text extraction."""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Optional

from services.vision.schemas import OCRBlock, OCRExtraction, OCRMeasurement


DEFAULT_DETECTION_MODEL = "PP-OCRv5_mobile_det"
DEFAULT_RECOGNITION_MODEL = "PP-OCRv5_mobile_rec"
SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

_MEASUREMENT_PATTERN = re.compile(
    r"(?<![\w.])"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>kWh|MWh|Wh|kW|MW|W|mV|kV|V|mA|A|mHz|kHz|Hz|%|°C)"
    r"\s*(?P<qualifier>AC|DC)?",
    flags=re.IGNORECASE,
)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    return float(value) if value else default


def _normalize_unit(unit: str) -> str:
    canonical = {
        "KWH": "kWh",
        "MWH": "MWh",
        "WH": "Wh",
        "KW": "kW",
        "MW": "MW",
        "W": "W",
        "MV": "mV",
        "KV": "kV",
        "V": "V",
        "MA": "mA",
        "A": "A",
        "MHZ": "mHz",
        "KHZ": "kHz",
        "HZ": "Hz",
        "%": "%",
        "°C": "°C",
    }
    return canonical[unit.upper()]


def extract_measurements(text: str) -> list[OCRMeasurement]:
    """Extract technical measurements without changing the source OCR text."""

    measurements: list[OCRMeasurement] = []
    seen: set[tuple[float, str, Optional[str]]] = set()
    for match in _MEASUREMENT_PATTERN.finditer(text):
        value = float(match.group("value"))
        unit = _normalize_unit(match.group("unit"))
        qualifier_value = match.group("qualifier")
        qualifier = qualifier_value.upper() if qualifier_value else None
        key = (value, unit, qualifier)
        if key in seen:
            continue
        seen.add(key)
        display_value = str(int(value)) if value.is_integer() else str(value)
        normalized = f"{display_value} {unit}"
        if qualifier:
            normalized += f" {qualifier}"
        measurements.append(
            OCRMeasurement(
                raw_text=match.group(0),
                value=value,
                unit=unit,
                qualifier=qualifier,
                normalized=normalized,
            )
        )
    return measurements


def _result_payload(item: Any) -> Mapping[str, Any]:
    value = getattr(item, "json", item)
    if callable(value):
        value = value()
    if not isinstance(value, Mapping):
        raise TypeError(f"Unsupported PaddleOCR result type: {type(item)!r}")
    payload = value.get("res", value)
    if not isinstance(payload, Mapping):
        raise TypeError("PaddleOCR result does not contain a mapping payload")
    return payload


class PaddleOCREngine:
    """Load PP-OCRv5 Mobile once and reuse it for CPU inference."""

    def __init__(
        self,
        *,
        detection_model: Optional[str] = None,
        recognition_model: Optional[str] = None,
        min_confidence: Optional[float] = None,
        pipeline_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.detection_model = detection_model or os.getenv(
            "VISION_OCR_DETECTION_MODEL", DEFAULT_DETECTION_MODEL
        )
        self.recognition_model = recognition_model or os.getenv(
            "VISION_OCR_RECOGNITION_MODEL", DEFAULT_RECOGNITION_MODEL
        )
        self.min_confidence = (
            min_confidence
            if min_confidence is not None
            else _env_float("VISION_OCR_MIN_CONFIDENCE", 0.5)
        )
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("OCR minimum confidence must be between 0 and 1")
        self._pipeline_factory = pipeline_factory
        self._pipeline: Any = None
        self._model_load_ms = 0.0
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def model_load_ms(self) -> float:
        return self._model_load_ms

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        with self._load_lock:
            if self._pipeline is not None:
                return self._pipeline
            started = perf_counter()
            if self._pipeline_factory is not None:
                pipeline = self._pipeline_factory()
            else:
                os.environ.setdefault(
                    "PADDLE_PDX_CACHE_HOME",
                    str(Path(__file__).resolve().parents[2] / ".cache" / "paddle"),
                )
                os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
                os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
                from paddleocr import PaddleOCR

                pipeline = PaddleOCR(
                    text_detection_model_name=self.detection_model,
                    text_recognition_model_name=self.recognition_model,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    device="cpu",
                )
            self._pipeline = pipeline
            self._model_load_ms = (perf_counter() - started) * 1000
        return self._pipeline

    def extract(self, image_path: str | Path) -> OCRExtraction:
        """Recognize one image and return serializable normalized data."""

        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image does not exist: {path}")
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image format: {path.suffix}")

        pipeline = self._load()
        started = perf_counter()
        with self._predict_lock:
            # PaddleOCR may return an iterator. Materialize it while the lock is
            # held so timing and thread-safety cover the complete inference run.
            raw_results = list(pipeline.predict(str(path)))
        inference_ms = (perf_counter() - started) * 1000

        blocks: list[OCRBlock] = []
        discarded = 0
        for item in raw_results:
            payload = _result_payload(item)
            texts = list(payload.get("rec_texts") or [])
            scores = list(payload.get("rec_scores") or [])
            polygons = payload.get("rec_polys")
            polygon_values = polygons.tolist() if hasattr(polygons, "tolist") else list(polygons or [])
            for index, raw_text in enumerate(texts):
                text = str(raw_text).strip()
                confidence = float(scores[index]) if index < len(scores) else 0.0
                if not text or confidence < self.min_confidence:
                    discarded += 1
                    continue
                raw_bbox = polygon_values[index] if index < len(polygon_values) else []
                bbox = [[int(value) for value in point] for point in raw_bbox]
                blocks.append(OCRBlock(text=text, confidence=confidence, bbox=bbox))

        full_text = "\n".join(block.text for block in blocks)
        return OCRExtraction(
            engine="paddleocr-cpu",
            detection_model=self.detection_model,
            recognition_model=self.recognition_model,
            image_path=str(path),
            full_text=full_text,
            blocks=blocks,
            measurements=extract_measurements(full_text),
            discarded_blocks=discarded,
            model_load_ms=self._model_load_ms,
            inference_ms=inference_ms,
        )
