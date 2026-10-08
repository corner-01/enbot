import unittest
from unittest.mock import Mock, patch

from demo import gradio_app


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class GradioDemoTests(unittest.TestCase):
    def test_format_sources_deduplicates_values(self):
        self.assertEqual(
            gradio_app.format_sources(["a", "a", {"url": "b"}]),
            "- a\n- b",
        )

    def test_ask_text_uses_existing_api_contract(self):
        response = _Response(
            {
                "status": "success",
                "request_id": "request-1",
                "data": {
                    "response": "answer",
                    "sources": ["manual.pdf"],
                    "retrieval_backend": "milvus",
                },
            }
        )
        with patch.object(gradio_app.HTTP, "post", Mock(return_value=response)):
            answer, sources, metadata = gradio_app.ask_text("question", "demo")
        self.assertEqual(answer, "answer")
        self.assertIn("manual.pdf", sources)
        self.assertEqual(metadata["retrieval_backend"], "milvus")

    def test_empty_question_does_not_call_api(self):
        with patch.object(gradio_app.HTTP, "post") as post:
            answer, _, metadata = gradio_app.ask_text("", "demo")
        self.assertIn("请输入", answer)
        self.assertEqual(metadata["status"], "waiting")
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
