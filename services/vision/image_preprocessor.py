"""Deterministic preprocessing for OCR and local vision-model inference."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from services.vision.image_validator import validate_image


DEFAULT_MAX_LONG_SIDE = 2048
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2] / "tmp" / "vision-preprocessed"
)


class ImagePreprocessingError(ValueError):
    """Raised when an image cannot safely enter the vision pipeline."""


@dataclass(frozen=True)
class ImagePreprocessingResult:
    """Paths and transformations produced for one source image."""

    original_path: str
    source_sha256: str
    vlm_image_path: str
    ocr_image_path: str
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    resized: bool
    exif_transformed: bool
    ocr_enhanced: bool
    ocr_contrast_before: float
    ocr_sharpness_before: float
    transformations: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _save_png_atomic(image: Image.Image, destination: Path) -> None:
    """Write a complete PNG before replacing a previous cached result."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".png",
            prefix=f".{destination.stem}-",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        image.save(temporary_path, format="PNG", optimize=True)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def preprocess_image(
    image_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    max_long_side: int = DEFAULT_MAX_LONG_SIDE,
    contrast_threshold: float = 25.0,
    sharpness_threshold: float = 100.0,
) -> ImagePreprocessingResult:
    """Create separate lossless inputs for the VLM and OCR engines.

    The VLM image preserves RGB information and layout. The OCR image is
    grayscale and only receives CLAHE or mild unsharp masking when needed.
    """

    if max_long_side < 256:
        raise ValueError("max_long_side must be at least 256 pixels")
    if contrast_threshold < 0 or sharpness_threshold < 0:
        raise ValueError("image quality thresholds cannot be negative")

    source_path = Path(image_path).expanduser().resolve()
    validation = validate_image(source_path)
    if not validation.accepted:
        details = "; ".join(validation.errors) or "unknown validation error"
        raise ImagePreprocessingError(f"Image validation failed: {details}")

    destination_dir = Path(output_dir or DEFAULT_OUTPUT_DIR).expanduser().resolve()
    source_sha256 = _sha256(source_path)
    cache_key = (
        f"{source_sha256[:16]}_l{max_long_side}"
        f"_c{_safe_token(contrast_threshold)}"
        f"_s{_safe_token(sharpness_threshold)}"
    )
    vlm_path = destination_dir / f"{cache_key}_vlm.png"
    ocr_path = destination_dir / f"{cache_key}_ocr.png"

    transformations: list[str] = ["validation_passed", "exif_orientation_checked"]
    with Image.open(source_path) as opened_image:
        orientation = int(opened_image.getexif().get(274, 1) or 1)
        exif_transformed = orientation not in (0, 1)
        rgb_image = ImageOps.exif_transpose(opened_image).convert("RGB")

    width, height = rgb_image.size
    longest_side = max(width, height)
    resized = longest_side > max_long_side
    if resized:
        scale = max_long_side / longest_side
        target_size = (
            max(1, round(width * scale)),
            max(1, round(height * scale)),
        )
        rgb_image = rgb_image.resize(target_size, Image.Resampling.LANCZOS)
        transformations.append(f"resized_to_{target_size[0]}x{target_size[1]}")

    processed_width, processed_height = rgb_image.size
    _save_png_atomic(rgb_image, vlm_path)
    transformations.append("vlm_rgb_png_created")

    rgb_array = np.asarray(rgb_image)
    grayscale = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    contrast_before = float(grayscale.std())
    sharpness_before = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
    ocr_enhanced = False

    if contrast_before < contrast_threshold:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        grayscale = clahe.apply(grayscale)
        transformations.append("ocr_clahe_applied")
        ocr_enhanced = True

    if sharpness_before < sharpness_threshold:
        blurred = cv2.GaussianBlur(grayscale, (0, 0), sigmaX=1.0)
        grayscale = cv2.addWeighted(grayscale, 1.5, blurred, -0.5, 0)
        transformations.append("ocr_mild_unsharp_applied")
        ocr_enhanced = True

    _save_png_atomic(Image.fromarray(grayscale, mode="L"), ocr_path)
    transformations.append("ocr_grayscale_png_created")

    return ImagePreprocessingResult(
        original_path=str(source_path),
        source_sha256=source_sha256,
        vlm_image_path=str(vlm_path),
        ocr_image_path=str(ocr_path),
        original_width=validation.width,
        original_height=validation.height,
        processed_width=processed_width,
        processed_height=processed_height,
        resized=resized,
        exif_transformed=exif_transformed,
        ocr_enhanced=ocr_enhanced,
        ocr_contrast_before=contrast_before,
        ocr_sharpness_before=sharpness_before,
        transformations=tuple(transformations),
    )
