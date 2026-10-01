"""Typed data structures shared by the multimodal vision service."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class OCRBlock:
    """One recognized text region."""

    text: str
    confidence: float
    bbox: list[list[int]] = field(default_factory=list)


@dataclass(frozen=True)
class OCRMeasurement:
    """A number and engineering unit extracted from OCR text."""

    raw_text: str
    value: float
    unit: str
    qualifier: Optional[str]
    normalized: str


@dataclass(frozen=True)
class OCRExtraction:
    """Normalized OCR output consumed by result fusion."""

    engine: str
    detection_model: str
    recognition_model: str
    image_path: str
    full_text: str
    blocks: list[OCRBlock]
    measurements: list[OCRMeasurement]
    discarded_blocks: int
    model_load_ms: float
    inference_ms: float


class VisualConnection(BaseModel):
    """One connection that is directly visible in a diagram or photograph."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source: str = Field(min_length=1, max_length=200)
    target: str = Field(min_length=1, max_length=200)
    label: Optional[str] = Field(default=None, max_length=300)


class VisualExtraction(BaseModel):
    """Strict visual evidence returned by the local multimodal model."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    image_type: Literal[
        "lcd_screen",
        "wiring_diagram",
        "nameplate",
        "app_screenshot",
        "installation_scene",
        "product_photo",
        "document_page",
        "unknown",
    ]
    manufacturer: Optional[str]
    product_model: Optional[str]
    visible_status: list[str] = Field(max_length=8)
    connections: list[VisualConnection] = Field(max_length=16)
    observations: list[str] = Field(max_length=10)
    possible_faults: list[str] = Field(max_length=6)
    out_of_domain: bool
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainties: list[str] = Field(max_length=8)


@dataclass(frozen=True)
class VisionInferenceResult:
    """Validated extraction plus Ollama performance metadata."""

    model: str
    prompt_version: str
    image_path: str
    extraction: VisualExtraction
    normalization_warnings: tuple[str, ...]
    wall_seconds: float
    total_seconds: float
    load_seconds: float
    prompt_tokens: int
    generated_tokens: int
    tokens_per_second: float

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "prompt_version": self.prompt_version,
            "image_path": self.image_path,
            "extraction": self.extraction.model_dump(mode="json"),
            "normalization_warnings": list(self.normalization_warnings),
            "wall_seconds": self.wall_seconds,
            "total_seconds": self.total_seconds,
            "load_seconds": self.load_seconds,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "tokens_per_second": self.tokens_per_second,
        }


@dataclass(frozen=True)
class FusedConnection:
    """A VLM connection annotated with OCR text support."""

    source: str
    target: str
    label: Optional[str]
    ocr_terms_present: bool


@dataclass(frozen=True)
class MultimodalFusionResult:
    """Deterministic evidence package consumed by retrieval and diagnosis."""

    image_type: str
    manufacturer: Optional[str]
    product_model: Optional[str]
    visible_status: tuple[str, ...]
    measurements: tuple[OCRMeasurement, ...]
    connections: tuple[FusedConnection, ...]
    observations: tuple[str, ...]
    possible_faults: tuple[str, ...]
    out_of_domain: bool
    retrieval_allowed: bool
    evidence_summary: str
    retrieval_query: str
    conflicts: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "image_type": self.image_type,
            "manufacturer": self.manufacturer,
            "product_model": self.product_model,
            "visible_status": list(self.visible_status),
            "measurements": [asdict(item) for item in self.measurements],
            "connections": [asdict(item) for item in self.connections],
            "observations": list(self.observations),
            "possible_faults": list(self.possible_faults),
            "out_of_domain": self.out_of_domain,
            "retrieval_allowed": self.retrieval_allowed,
            "evidence_summary": self.evidence_summary,
            "retrieval_query": self.retrieval_query,
            "conflicts": list(self.conflicts),
            "warnings": list(self.warnings),
        }
