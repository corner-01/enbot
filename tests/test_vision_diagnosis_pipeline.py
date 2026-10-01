import importlib.util
import json
import unittest

import httpx


VISION_DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("cv2", "numpy", "PIL")
)

if VISION_DEPENDENCIES_AVAILABLE:
    from services.vision.diagnosis_pipeline import (
        EnBotAPIResponseError,
        MultimodalDiagnosisPipeline,
    )
    from services.vision.image_preprocessor import ImagePreprocessingResult
    from services.vision.schemas import (
        OCRBlock,
        OCRExtraction,
        VisionInferenceResult,
        VisualExtraction,
    )


def _preprocessing():
    return ImagePreprocessingResult(
        original_path="source.png",
        source_sha256="a" * 64,
        vlm_image_path="vlm.png",
        ocr_image_path="ocr.png",
        original_width=1000,
        original_height=800,
        processed_width=1000,
        processed_height=800,
        resized=False,
        exif_transformed=False,
        ocr_enhanced=False,
        ocr_contrast_before=50.0,
        ocr_sharpness_before=500.0,
        transformations=("validation_passed",),
    )


def _ocr(text: str):
    return OCRExtraction(
        engine="paddleocr-cpu",
        detection_model="det",
        recognition_model="rec",
        image_path="ocr.png",
        full_text=text,
        blocks=[OCRBlock(text=text, confidence=0.99)],
        measurements=[],
        discarded_blocks=0,
        model_load_ms=0.0,
        inference_ms=1.0,
    )


def _vision(manufacturer: str, out_of_domain: bool):
    return VisionInferenceResult(
        model="qwen3-vl:2b-instruct",
        prompt_version="visual-extraction-v1",
        image_path="vlm.png",
        extraction=VisualExtraction(
            image_type="product_photo",
            manufacturer=manufacturer,
            product_model=None,
            visible_status=[],
            connections=[],
            observations=["A product is visible."],
            possible_faults=[],
            out_of_domain=out_of_domain,
            confidence=0.9,
            uncertainties=[],
        ),
        normalization_warnings=(),
        wall_seconds=1.0,
        total_seconds=1.0,
        load_seconds=0.0,
        prompt_tokens=100,
        generated_tokens=20,
        tokens_per_second=20.0,
    )


class _FakeOCREngine:
    def __init__(self, result):
        self.result = result
        self.paths = []

    def extract(self, path):
        self.paths.append(path)
        return self.result


class _FakeVisionClient:
    def __init__(self, result):
        self.result = result
        self.paths = []

    def analyze(self, path):
        self.paths.append(path)
        return self.result


@unittest.skipUnless(
    VISION_DEPENDENCIES_AVAILABLE,
    "vision pipeline dependencies are installed in services/vision/.venv",
)
class VisionDiagnosisPipelineTests(unittest.TestCase):
    def test_in_domain_result_calls_existing_chat_api(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "status": "success",
                    "request_id": "request-1",
                    "data": {
                        "response": "Grounded answer",
                        "sources": ["https://example.test/manual"],
                        "retrieval_backend": "milvus",
                    },
                },
            )

        with httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url="http://enbot.test",
        ) as http_client:
            pipeline = MultimodalDiagnosisPipeline(
                preprocessor=lambda _path: _preprocessing(),
                ocr_engine=_FakeOCREngine(_ocr("HANCHU ESS inverter")),
                vision_client=_FakeVisionClient(_vision("Hanchu ESS", False)),
                http_client=http_client,
            )
            result = pipeline.run(
                "source.png",
                question="What is shown?",
                session_id="vision-test",
            )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.answer, "Grounded answer")
        self.assertEqual(result.request_id, "request-1")
        self.assertEqual(result.retrieval_backend, "milvus")
        self.assertIn("What is shown?", captured["query"])
        self.assertEqual(captured["session_id"], "vision-test")
        self.assertFalse(captured["evaluate"])
        self.assertFalse(captured["debug"])

    def test_out_of_domain_result_never_calls_enbot_api(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            self.fail("EnBot API must not be called for an out-of-domain image")

        with httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url="http://enbot.test",
        ) as http_client:
            pipeline = MultimodalDiagnosisPipeline(
                preprocessor=lambda _path: _preprocessing(),
                ocr_engine=_FakeOCREngine(_ocr("External Energy Ltd.")),
                vision_client=_FakeVisionClient(_vision("External Energy Ltd.", True)),
                http_client=http_client,
            )
            result = pipeline.run("source.png", question="Diagnose this")

        self.assertEqual(result.status, "out_of_domain")
        self.assertFalse(result.fusion.retrieval_allowed)
        self.assertEqual(result.fusion.retrieval_query, "")
        self.assertIsNone(result.request_id)
        self.assertEqual(result.stage_seconds["rag"], 0.0)

    def test_invalid_enbot_response_is_rejected(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"status": "success"})

        with httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url="http://enbot.test",
        ) as http_client:
            pipeline = MultimodalDiagnosisPipeline(
                preprocessor=lambda _path: _preprocessing(),
                ocr_engine=_FakeOCREngine(_ocr("EnValt inverter")),
                vision_client=_FakeVisionClient(_vision("EnValt", False)),
                http_client=http_client,
            )
            with self.assertRaises(EnBotAPIResponseError):
                pipeline.run("source.png")

    def test_rejects_unsafe_session_id(self):
        pipeline = MultimodalDiagnosisPipeline(
            preprocessor=lambda _path: _preprocessing(),
            ocr_engine=_FakeOCREngine(_ocr("EnValt")),
            vision_client=_FakeVisionClient(_vision("EnValt", False)),
            http_client=httpx.Client(base_url="http://enbot.test"),
        )
        try:
            with self.assertRaises(ValueError):
                pipeline.run("source.png", session_id="unsafe session")
        finally:
            pipeline._http_client.close()


if __name__ == "__main__":
    unittest.main()
