import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SERVICE_DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(module_name) is not None
    for module_name in ("cv2", "fastapi", "multipart")
)

if SERVICE_DEPENDENCIES_AVAILABLE:
    from fastapi.testclient import TestClient

    import services.vision.app as vision_app


class _FakeResult:
    status = "out_of_domain"

    def to_dict(self, *, include_debug=False):
        result = {
            "status": self.status,
            "answer": "Safe refusal",
            "request_id": None,
            "sources": [],
            "retrieval_backend": None,
            "fusion": {"retrieval_allowed": False},
            "total_seconds": 0.01,
            "stage_seconds": {"rag": 0.0},
        }
        if include_debug:
            result["ocr"] = {"full_text": "EXTERNAL"}
        return result


class _FakeRuntime:
    def __init__(self):
        self.seen_paths = []

    async def diagnose(self, image_path, *, question, session_id):
        self.seen_paths.append(Path(image_path))
        assert Path(image_path).is_file()
        assert question == "What is shown?"
        assert session_id == "api-test"
        return _FakeResult()

    async def close(self):
        return None

    def metrics(self):
        return {"initialized": False, "requests_total": len(self.seen_paths)}


@unittest.skipUnless(
    SERVICE_DEPENDENCIES_AVAILABLE,
    "vision API dependencies are installed in services/vision/.venv",
)
class VisionServiceAPITests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.fake_runtime = _FakeRuntime()
        self.env = patch.dict(
            os.environ,
            {
                "VISION_UPLOAD_DIR": self.temp_dir.name,
                "VISION_MAX_IMAGE_MIB": "1",
                "VISION_ENABLE_DEBUG": "False",
                "VISION_API_KEY": "",
            },
            clear=False,
        )
        self.env.start()
        self.runtime_patch = patch.object(
            vision_app,
            "runtime",
            self.fake_runtime,
        )
        self.runtime_patch.start()
        self.client = TestClient(vision_app.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.runtime_patch.stop()
        self.env.stop()
        self.temp_dir.cleanup()

    def test_liveness_does_not_initialize_pipeline(self):
        response = self.client.get("/health/live")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["service"], "enbot-vision")
        self.assertEqual(self.fake_runtime.seen_paths, [])

    def test_multipart_diagnosis_returns_contract_and_deletes_upload(self):
        response = self.client.post(
            "/v1/diagnose",
            files={"image": ("device.png", b"not-decoded-by-fake", "image/png")},
            data={
                "question": "What is shown?",
                "session_id": "api-test",
                "debug": "false",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "out_of_domain")
        self.assertTrue(payload["request_id"])
        self.assertEqual(payload["data"]["answer"], "Safe refusal")
        self.assertEqual(len(self.fake_runtime.seen_paths), 1)
        self.assertFalse(self.fake_runtime.seen_paths[0].exists())

    def test_unsupported_media_type_is_rejected(self):
        response = self.client.post(
            "/v1/diagnose",
            files={"image": ("device.txt", b"text", "text/plain")},
        )
        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["status"], "error")

    def test_oversized_upload_is_rejected_and_deleted(self):
        with patch.dict(os.environ, {"VISION_MAX_IMAGE_MIB": "0.000001"}):
            response = self.client.post(
                "/v1/diagnose",
                files={"image": ("device.png", b"too-large", "image/png")},
            )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(list(Path(self.temp_dir.name).glob("upload-*")), [])

    def test_debug_requires_explicit_server_setting(self):
        response = self.client.post(
            "/v1/diagnose",
            files={"image": ("device.png", b"image", "image/png")},
            data={"debug": "true"},
        )
        self.assertEqual(response.status_code, 403)

    def test_api_key_is_enforced_when_configured(self):
        with patch.dict(os.environ, {"VISION_API_KEY": "secret"}):
            denied = self.client.get("/metrics")
            allowed = self.client.get("/metrics", headers={"X-API-Key": "secret"})
        self.assertEqual(denied.status_code, 401)
        self.assertEqual(allowed.status_code, 200)


if __name__ == "__main__":
    unittest.main()
