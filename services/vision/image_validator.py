"""Validate images before OCR and vision-model inference."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}


@dataclass(frozen=True)
class ImageValidationResult:
    accepted: bool
    image_path: str
    format: str | None = None
    width: int = 0
    height: int = 0
    file_size_bytes: int = 0
    sharpness: float | None = None
    brightness: float | None = None
    contrast: float | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def validate_image(
    image_path: str | Path,
    *,
    max_image_mib: float | None = None,
    min_side: int = 96,
    max_pixels: int = 40_000_000,
    blurry_threshold: float = 40.0,
) -> ImageValidationResult:
    """Validate file integrity, size, dimensions and basic image quality."""

    path = Path(image_path).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []

    if not path.is_file():
        return ImageValidationResult(
            accepted=False,
            image_path=str(path),
            errors=["图片文件不存在"],
        )

    file_size = path.stat().st_size
    size_limit_mib = max_image_mib or float(
        os.getenv("VISION_MAX_IMAGE_MIB", "10")
    )
    size_limit_bytes = int(size_limit_mib * 1024 * 1024)

    if file_size == 0:
        errors.append("图片文件为空")

    if file_size > size_limit_bytes:
        errors.append(
            f"图片超过大小限制：{file_size / 1024 / 1024:.2f} MiB，"
            f"最大允许 {size_limit_mib:.2f} MiB"
        )

    image_format: str | None = None
    width = 0
    height = 0

    try:
        # verify() 用来检测损坏、伪造或无法完整解码的图片。
        with Image.open(path) as image:
            image_format = image.format
            width, height = image.size
            image.verify()

        # 再次打开并真正解码像素，避免只检查文件头。
        with Image.open(path) as image:
            image.load()

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        errors.append(f"图片无法正常解码：{exc}")

    if image_format and image_format.upper() not in SUPPORTED_FORMATS:
        errors.append(f"不支持的图片格式：{image_format}")

    if width and height:
        if min(width, height) < min_side:
            errors.append(
                f"图片尺寸过小：{width}×{height}，短边至少需要 {min_side} 像素"
            )

        if width * height > max_pixels:
            errors.append(
                f"图片像素过多：{width * height:,}，"
                f"最大允许 {max_pixels:,}"
            )

    sharpness: float | None = None
    brightness: float | None = None
    contrast: float | None = None

    if not errors:
        image_bytes = np.fromfile(str(path), dtype=np.uint8)
        decoded = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if decoded is None:
            errors.append("OpenCV 无法解码图片")
        else:
            gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness = float(gray.mean())
            contrast = float(gray.std())

            # 清晰度只产生警告，不直接拒绝。
            # 某些正常的液晶屏或纯色界面本身边缘较少。
            if sharpness < blurry_threshold:
                warnings.append(
                    f"图片可能模糊，清晰度得分为 {sharpness:.2f}"
                )

            if brightness < 25:
                warnings.append(
                    f"图片可能过暗，平均亮度为 {brightness:.2f}"
                )
            elif brightness > 245 and contrast < 15:
                warnings.append(
                    f"图片可能过亮，平均亮度为 {brightness:.2f}，"
                    f"对比度为 {contrast:.2f}"
                )

    return ImageValidationResult(
        accepted=not errors,
        image_path=str(path),
        format=image_format,
        width=width,
        height=height,
        file_size_bytes=file_size,
        sharpness=sharpness,
        brightness=brightness,
        contrast=contrast,
        warnings=warnings,
        errors=errors,
    )
