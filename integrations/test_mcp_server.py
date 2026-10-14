import unittest
from unittest.mock import Mock, patch

from integrations.mcp_server import EnBotApiClient, EnBotApiError


class EnBotMcpApiClientTests(unittest.TestCase):
    def setUp(self):
        self.client = EnBotApiClient(
            base_url="http://127.0.0.1:8000/",
            api_key="test-key",
            timeout_seconds=12,
        )

    @patch("integrations.mcp_server.requests.request")
    def test_ask_forwards_payload_and_api_key(self, request):
        response = Mock(ok=True)
        response.json.return_value = {
            "status": "success",
            "request_id": "request-1",
            "data": {"response": "回答", "sources": ["manual.pdf"]},
        }
        request.return_value = response

        result = self.client.ask("如何处理告警？", "mcp-user")

        self.assertEqual(result["request_id"], "request-1")
        request.assert_called_once_with(
            "POST",
            "http://127.0.0.1:8000/v1/chat",
            json={
                "query": "如何处理告警？",
                "session_id": "mcp-user",
                "debug": False,
            },
            headers={"Accept": "application/json", "X-API-Key": "test-key"},
            timeout=12,
        )

    @patch("integrations.mcp_server.requests.request")
    def test_retrieve_never_requests_private_document_content(self, request):
        response = Mock(ok=True)
        response.json.return_value = {
            "status": "success",
            "request_id": "request-2",
            "data": {"retrieval_backend": "milvus", "retrieval_debug": []},
        }
        request.return_value = response

        self.client.retrieve("逆变器故障码")

        self.assertFalse(request.call_args.kwargs["json"]["include_content"])

    @patch("integrations.mcp_server.requests.request")
    def test_feedback_forwards_the_original_request_id(self, request):
        response = Mock(ok=True)
        response.json.return_value = {"status": "success", "data": {}}
        request.return_value = response

        self.client.feedback(
            "request-1",
            "down",
            categories=["回答不完整"],
            comment="缺少复位步骤",
            expected_answer="先断电检查，再按手册复位。",
        )

        self.assertEqual(request.call_args.kwargs["json"]["request_id"], "request-1")
        self.assertEqual(request.call_args.kwargs["json"]["rating"], "down")

    @patch("integrations.mcp_server.requests.request")
    def test_http_error_is_reported_as_tool_friendly_error(self, request):
        response = Mock(ok=False, status_code=503)
        response.json.return_value = {"detail": "model unavailable"}
        request.return_value = response

        with self.assertRaisesRegex(EnBotApiError, "HTTP 503"):
            self.client.health()


if __name__ == "__main__":
    unittest.main()
