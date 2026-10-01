import base64
import json
import tempfile
import unittest
from pathlib import Path

import httpx

from services.vision.vision_client import (
    OllamaVisionClient,
    VisionModelResponseError,
    VisionServiceUnavailableError,
)


def _valid_extraction() -> dict:
    return {
        "image_type": "wiring_diagram",
        "manufacturer": "Example Energy",
        "product_model": None,
        "visible_status": [],
        "connections": [
            {"source": "PV", "target": "Inverter", "label": "PV+ / PV-"}
        ],
        "observations": ["A photovoltaic input is visible."],
        "possible_faults": [],
        "out_of_domain": True,
        "confidence": 0.9,
        "uncertainties": ["No EnValt or Hanchu model is visible."],
    }


class VisionClientTests(unittest.TestCase):
    def test_sends_image_and_schema_and_validates_result(self):
        captured_payload = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_payload.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "model": "qwen3-vl:2b-instruct",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(_valid_extraction()),
                    },
                    "total_duration": 2_000_000_000,
                    "load_duration": 100_000_000,
                    "prompt_eval_count": 100,
                    "eval_count": 50,
                    "eval_duration": 1_000_000_000,
                },
            )

        transport = httpx.MockTransport(handler)
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "image.png"
            image_path.write_bytes(b"fake-image")
            with httpx.Client(
                transport=transport,
                base_url="http://ollama.test",
            ) as http_client:
                client = OllamaVisionClient(http_client=http_client)
                result = client.analyze(image_path)

        self.assertEqual(result.extraction.image_type, "wiring_diagram")
        self.assertTrue(result.extraction.out_of_domain)
        self.assertEqual(result.tokens_per_second, 50.0)
        self.assertFalse(captured_payload["stream"])
        self.assertEqual(captured_payload["options"]["temperature"], 0)
        self.assertEqual(
            base64.b64decode(captured_payload["messages"][0]["images"][0]),
            b"fake-image",
        )
        self.assertEqual(captured_payload["format"]["type"], "object")

    def test_rejects_model_output_that_does_not_match_schema(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": '{"image_type":"wiring_diagram"}',
                    }
                },
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "image.png"
            image_path.write_bytes(b"fake-image")
            with httpx.Client(
                transport=httpx.MockTransport(handler),
                base_url="http://ollama.test",
            ) as http_client:
                client = OllamaVisionClient(http_client=http_client)
                with self.assertRaises(VisionModelResponseError):
                    client.analyze(image_path)

    def test_applies_conservative_domain_and_fault_guards(self):
        model_output = _valid_extraction()
        model_output["out_of_domain"] = False
        model_output["possible_faults"] = [
            "The connection is not clearly visible.",
            "Fault E101 is displayed.",
        ]

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(model_output),
                    }
                },
            )

        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "image.png"
            image_path.write_bytes(b"fake-image")
            with httpx.Client(
                transport=httpx.MockTransport(handler),
                base_url="http://ollama.test",
            ) as http_client:
                result = OllamaVisionClient(http_client=http_client).analyze(image_path)

        self.assertTrue(result.extraction.out_of_domain)
        self.assertEqual(result.extraction.possible_faults, ["Fault E101 is displayed."])
        self.assertEqual(len(result.normalization_warnings), 2)

    def test_reports_connection_failure(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "image.png"
            image_path.write_bytes(b"fake-image")
            with httpx.Client(
                transport=httpx.MockTransport(handler),
                base_url="http://ollama.test",
            ) as http_client:
                client = OllamaVisionClient(http_client=http_client)
                with self.assertRaises(VisionServiceUnavailableError):
                    client.analyze(image_path)

    def test_rejects_missing_image_before_request(self):
        with OllamaVisionClient() as client:
            with self.assertRaises(FileNotFoundError):
                client.analyze("missing-vision-image.png")


if __name__ == "__main__":
    unittest.main()
