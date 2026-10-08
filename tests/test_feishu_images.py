import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from integrations.feishu_bot import (
    FeishuConfig,
    FeishuEnBotBridge,
    MessageJob,
    detect_image_media_type,
    extract_image_key,
    extract_media_reference,
    extract_vision_answer,
)


PNG = b"\x89PNG\r\n\x1a\n" + b"test-image"


class _ResourceResponse:
    code = 0
    msg = "ok"
    file_name = None

    def __init__(self, content=PNG):
        self.file = io.BytesIO(content)

    def success(self):
        return True

    def get_log_id(self):
        return "test-log"


class _HTTPResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "status": "success",
            "request_id": "vision-1",
            "data": {
                "answer": "检测到 BMS 通信告警。",
                "sources": ["manual.pdf#page=9"],
            },
        }


def _config():
    with patch.dict(
        "os.environ",
        {
            "FEISHU_VISION_ENABLED": "true",
            "FEISHU_VISION_API_URL": "http://127.0.0.1:8011/v1/diagnose",
            "VISION_API_KEY": "vision-secret",
        },
        clear=False,
    ):
        return FeishuConfig.from_env(require_credentials=False)


class FeishuImageTests(unittest.TestCase):
    def test_extract_image_key(self):
        self.assertEqual(extract_image_key('{"image_key":"img_123"}'), "img_123")
        self.assertEqual(extract_image_key("not-json"), "")

    def test_extracts_file_and_rich_post_images(self):
        self.assertEqual(
            extract_media_reference('{"file_key":"file_123"}', "file"),
            ("file_123", "file"),
        )
        post = '{"content":[[{"tag":"img","image_key":"img_post"}]]}'
        self.assertEqual(
            extract_media_reference(post, "post"),
            ("img_post", "image"),
        )

    def test_detect_image_media_type(self):
        self.assertEqual(detect_image_media_type(PNG), (".png", "image/png"))
        with self.assertRaises(ValueError):
            detect_image_media_type(b"not-an-image")

    def test_extract_vision_answer_appends_sources(self):
        answer = extract_vision_answer(_HTTPResponse().json())
        self.assertIn("BMS 通信告警", answer)
        self.assertIn("manual.pdf#page=9", answer)

    def test_downloads_feishu_image_and_calls_vision_api(self):
        resource = Mock()
        resource.get.return_value = _ResourceResponse()
        api_client = SimpleNamespace(
            im=SimpleNamespace(
                v1=SimpleNamespace(message_resource=resource),
            )
        )
        bridge = FeishuEnBotBridge(_config(), api_client)
        bridge.http.post = Mock(return_value=_HTTPResponse())

        answer = bridge._diagnose_image(
            MessageJob(
                message_id="message-1",
                chat_id="chat-1",
                message_type="image",
                image_key="img-1",
                resource_type="image",
            )
        )

        self.assertIn("BMS 通信告警", answer)
        resource.get.assert_called_once()
        _, kwargs = bridge.http.post.call_args
        self.assertEqual(kwargs["headers"], {"X-API-Key": "vision-secret"})
        self.assertEqual(kwargs["data"]["session_id"], "chat-1")
        self.assertEqual(kwargs["files"]["image"][2], "image/png")


if __name__ == "__main__":
    unittest.main()
