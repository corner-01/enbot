import tempfile
import unittest
from pathlib import Path

from services.vision.ocr_engine import PaddleOCREngine, extract_measurements


class _FakeResult:
    json = {
        "res": {
            "rec_texts": ["18A Breaker", "180ADC", "noise"],
            "rec_scores": [0.99, 0.98, 0.2],
            "rec_polys": [
                [[0, 0], [10, 0], [10, 10], [0, 10]],
                [[0, 20], [20, 20], [20, 30], [0, 30]],
                [[0, 40], [20, 40], [20, 50], [0, 50]],
            ],
        }
    }


class _FakePipeline:
    def __init__(self):
        self.calls = 0

    def predict(self, _image_path):
        self.calls += 1
        return [_FakeResult()]


class VisionOCREngineTests(unittest.TestCase):
    def test_measurements_are_normalized_and_deduplicated(self):
        values = extract_measurements("18A 32A 180ADC and 18 A")
        self.assertEqual(
            [item.normalized for item in values],
            ["18 A", "32 A", "180 A DC"],
        )

    def test_pipeline_is_lazy_reused_and_filters_low_confidence(self):
        pipeline = _FakePipeline()
        factory_calls = 0

        def factory():
            nonlocal factory_calls
            factory_calls += 1
            return pipeline

        with tempfile.NamedTemporaryFile(suffix=".png") as handle:
            engine = PaddleOCREngine(
                min_confidence=0.5,
                pipeline_factory=factory,
            )
            self.assertFalse(engine.loaded)
            first = engine.extract(Path(handle.name))
            second = engine.extract(Path(handle.name))

        self.assertTrue(engine.loaded)
        self.assertEqual(factory_calls, 1)
        self.assertEqual(pipeline.calls, 2)
        self.assertEqual([block.text for block in first.blocks], ["18A Breaker", "180ADC"])
        self.assertEqual(first.discarded_blocks, 1)
        self.assertEqual(
            [item.normalized for item in second.measurements],
            ["18 A", "180 A DC"],
        )


if __name__ == "__main__":
    unittest.main()
