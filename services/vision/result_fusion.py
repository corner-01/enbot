"""Deterministic fusion of exact OCR facts and semantic VLM evidence."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Optional

from services.vision.schemas import (
    FusedConnection,
    MultimodalFusionResult,
    OCRExtraction,
    VisionInferenceResult,
    VisualConnection,
)


_DOMAIN_BRANDS = {
    "envalt": "EnValt",
    "hanchu": "Hanchu",
}
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


def _lookup_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _is_present(ocr_lookup: str, value: Optional[str]) -> bool:
    if not value:
        return False
    candidate = _lookup_text(value)
    return bool(candidate) and candidate in ocr_lookup


def _deduplicate_strings(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        value = raw_value.strip()
        key = _lookup_text(value)
        if not value or not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return tuple(result)


def _detect_domain_brand(ocr_lookup: str) -> Optional[str]:
    for marker, display_name in _DOMAIN_BRANDS.items():
        if marker in ocr_lookup:
            return display_name
    return None


def _fault_has_ocr_support(fault: str, ocr_text: str, ocr_lookup: str) -> bool:
    codes = _FAULT_CODE_PATTERN.findall(fault)
    if codes:
        return any(_lookup_text(code) in ocr_lookup for code in codes)
    ocr_casefold = ocr_text.casefold()
    return any(
        term in fault.casefold() and term in ocr_casefold for term in _FAULT_TERMS
    )


def _fuse_connections(
    connections: Iterable[VisualConnection],
    ocr_lookup: str,
) -> tuple[FusedConnection, ...]:
    fused: list[FusedConnection] = []
    seen: set[tuple[str, str, str]] = set()
    for connection in connections:
        label = connection.label.strip() if connection.label else None
        key = (
            _lookup_text(connection.source),
            _lookup_text(connection.target),
            _lookup_text(label or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        terms_present = (
            _is_present(ocr_lookup, connection.source)
            and _is_present(ocr_lookup, connection.target)
            and (label is None or _is_present(ocr_lookup, label))
        )
        fused.append(
            FusedConnection(
                source=connection.source,
                target=connection.target,
                label=label,
                ocr_terms_present=terms_present,
            )
        )
    return tuple(fused)


def _build_evidence_summary(
    *,
    image_type: str,
    manufacturer: Optional[str],
    product_model: Optional[str],
    visible_status: tuple[str, ...],
    measurements: tuple,
    possible_faults: tuple[str, ...],
    connections: tuple[FusedConnection, ...],
) -> str:
    parts = [f"image_type={image_type}"]
    if manufacturer:
        parts.append(f"manufacturer={manufacturer}")
    if product_model:
        parts.append(f"product_model={product_model}")
    if visible_status:
        parts.append("visible_status=" + "; ".join(visible_status))
    if measurements:
        parts.append(
            "ocr_measurements=" + ", ".join(item.normalized for item in measurements)
        )
    if possible_faults:
        parts.append("explicit_faults=" + "; ".join(possible_faults))
    supported_connections = [item for item in connections if item.ocr_terms_present]
    if supported_connections:
        component_names: list[str] = []
        for item in supported_connections[:8]:
            component_names.extend([item.source, item.target])
            if item.label:
                component_names.append(item.label)
        parts.append(
            "ocr_supported_component_terms="
            + ", ".join(_deduplicate_strings(component_names))
        )
    return " | ".join(parts)


def fuse_multimodal_results(
    ocr: OCRExtraction,
    vision: VisionInferenceResult,
    *,
    user_question: Optional[str] = None,
) -> MultimodalFusionResult:
    """Fuse two independent evidence sources without allowing VLM fact repair."""

    ocr_text = ocr.full_text
    ocr_lookup = _lookup_text(ocr_text)
    visual = vision.extraction
    conflicts: list[str] = []
    warnings = list(vision.normalization_warnings)

    ocr_domain_brand = _detect_domain_brand(ocr_lookup)
    visual_manufacturer = visual.manufacturer.strip() if visual.manufacturer else None
    manufacturer = visual_manufacturer
    manufacturer_verified = bool(
        visual_manufacturer and _is_present(ocr_lookup, visual_manufacturer)
    )

    if ocr_domain_brand:
        if (
            visual_manufacturer
            and ocr_domain_brand.casefold() not in visual_manufacturer.casefold()
        ):
            conflicts.append(
                "VLM manufacturer conflicts with an EnValt/Hanchu brand visible in OCR: "
                f"VLM={visual_manufacturer}, OCR={ocr_domain_brand}"
            )
        manufacturer = ocr_domain_brand
        manufacturer_verified = True
    elif visual_manufacturer and not manufacturer_verified:
        conflicts.append(
            f"VLM manufacturer is not supported by OCR text: {visual_manufacturer}"
        )
        manufacturer = None

    visual_model = visual.product_model.strip() if visual.product_model else None
    if visual_model and _is_present(ocr_lookup, visual_model):
        product_model = visual_model
    else:
        product_model = None
        if visual_model:
            conflicts.append(f"VLM product model is not supported by OCR: {visual_model}")

    if ocr_domain_brand:
        out_of_domain = False
    elif visual_manufacturer and manufacturer_verified:
        out_of_domain = not any(
            marker in _lookup_text(visual_manufacturer) for marker in _DOMAIN_BRANDS
        )
    else:
        out_of_domain = visual.out_of_domain

    supported_faults: list[str] = []
    for fault in visual.possible_faults:
        if _fault_has_ocr_support(fault, ocr_text, ocr_lookup):
            supported_faults.append(fault)
        else:
            conflicts.append(f"VLM fault has no OCR evidence and was removed: {fault}")

    connections = _fuse_connections(visual.connections, ocr_lookup)
    unsupported_connection_count = sum(
        1 for connection in connections if not connection.ocr_terms_present
    )
    if unsupported_connection_count:
        warnings.append(
            f"{unsupported_connection_count} VLM connection(s) lack complete OCR term "
            "support and remain unverified."
        )
    if ocr.discarded_blocks:
        warnings.append(
            f"OCR discarded {ocr.discarded_blocks} low-confidence text block(s)."
        )

    visible_status = _deduplicate_strings(visual.visible_status)
    observations = _deduplicate_strings(visual.observations)
    possible_faults = _deduplicate_strings(supported_faults)
    measurements = tuple(ocr.measurements)
    evidence_summary = _build_evidence_summary(
        image_type=visual.image_type,
        manufacturer=manufacturer,
        product_model=product_model,
        visible_status=visible_status,
        measurements=measurements,
        possible_faults=possible_faults,
        connections=connections,
    )
    question = (user_question or "").strip()
    if out_of_domain:
        retrieval_query = ""
    else:
        retrieval_query = (
            f"question={question} | {evidence_summary}" if question else evidence_summary
        )
        retrieval_query = retrieval_query[:2000]

    return MultimodalFusionResult(
        image_type=visual.image_type,
        manufacturer=manufacturer,
        product_model=product_model,
        visible_status=visible_status,
        measurements=measurements,
        connections=connections,
        observations=observations,
        possible_faults=possible_faults,
        out_of_domain=out_of_domain,
        retrieval_allowed=not out_of_domain,
        evidence_summary=evidence_summary,
        retrieval_query=retrieval_query,
        conflicts=_deduplicate_strings(conflicts),
        warnings=_deduplicate_strings(warnings),
    )
