import importlib.util
import tempfile
import unittest
from pathlib import Path

VISION_DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("cv2", "numpy", "PIL")
)

if VISION_DEPENDENCIES_AVAILABLE:
    from PIL import Image, ImageDraw

    from services.vision.image_preprocessor import (
        ImagePreprocessingError,
        preprocess_image,
    )


@unittest.skipUnless(
    VISION_DEPENDENCIES_AVAILABLE,
    "vision preprocessing dependencies are installed in services/vision/.venv",
)
class VisionImagePreprocessorTests(unittest.TestCase):
    def test_creates_separate_vlm_and_ocr_images_without_upscaling(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.png"
            output = root / "output"
            image = Image.new("RGB", (320, 200), "white")
            draw = ImageDraw.Draw(image)
            draw.line((20, 40, 300, 40), fill="black", width=3)
            draw.rectangle((80, 70, 240, 150), outline="red", width=3)
            image.save(source)

            result = preprocess_image(source, output_dir=output)

            self.assertFalse(result.resized)
            self.assertEqual((result.processed_width, result.processed_height), (320, 200))
            self.assertTrue(Path(result.vlm_image_path).is_file())
            self.assertTrue(Path(result.ocr_image_path).is_file())
            with Image.open(result.vlm_image_path) as vlm_image:
                self.assertEqual(vlm_image.mode, "RGB")
            with Image.open(result.ocr_image_path) as ocr_image:
                self.assertEqual(ocr_image.mode, "L")

    def test_resizes_large_image_without_changing_aspect_ratio(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "large.png"
            Image.new("RGB", (800, 400), "gray").save(source)

            result = preprocess_image(
                source,
                output_dir=root / "output",
                max_long_side=400,
            )

            self.assertTrue(result.resized)
            self.assertEqual((result.processed_width, result.processed_height), (400, 200))

    def test_rejects_image_that_fails_validation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "too-small.png"
            Image.new("RGB", (32, 32), "white").save(source)

            with self.assertRaises(ImagePreprocessingError):
                preprocess_image(source, output_dir=root / "output")


if __name__ == "__main__":
    unittest.main()
