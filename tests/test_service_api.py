import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from pydantic import ValidationError

from app.service_api import (
    ChatRequest,
    ChatData,
    ChatResponse,
    FeedbackRequest,
    HelpfulFeedbackRequest,
    RetrieveRequest,
    UnhelpfulFeedbackRequest,
    _chat_sse_events,
    _sse_event,
    app,
    health_live,
    health_ready,
    require_api_key,
    service_metrics,
    submit_helpful_feedback,
    submit_unhelpful_feedback,
)
from app.feedback_store import FeedbackStore


class ServiceApiTests(unittest.TestCase):
    def test_chat_request_rejects_blank_query_and_unsafe_session_id(self):
        with self.assertRaises(ValidationError):
            ChatRequest(query="   ")
        with self.assertRaises(ValidationError):
            ChatRequest(query="hello", session_id="../unsafe")

    def test_request_models_trim_queries_and_forbid_extra_fields(self):
        self.assertEqual(ChatRequest(query="  hello  ").query, "hello")
        self.assertEqual(RetrieveRequest(query="  inverter  ").query, "inverter")
        with self.assertRaises(ValidationError):
            ChatRequest(query="hello", unexpected=True)

    def test_feedback_request_validates_rating_and_categories(self):
        payload = FeedbackRequest(
            request_id="abc123",
            rating="down",
            categories=["回答不完整", "回答不完整"],
            comment="  missing steps  ",
        )
        self.assertEqual(payload.categories, ["回答不完整"])
        self.assertEqual(payload.comment, "missing steps")
        with self.assertRaises(ValidationError):
            FeedbackRequest(
                request_id="abc123",
                rating="down",
                categories=["unsupported"],
            )

    def test_frontend_feedback_models_keep_two_button_contract_simple(self):
        helpful = HelpfulFeedbackRequest(request_id="request-1")
        unhelpful = UnhelpfulFeedbackRequest(
            request_id="request-1",
            categories=["回答不完整", "回答不完整"],
            comment="  缺少操作步骤  ",
        )

        self.assertEqual(helpful.request_id, "request-1")
        self.assertEqual(unhelpful.categories, ["回答不完整"])
        self.assertEqual(unhelpful.comment, "缺少操作步骤")

    def test_frontend_feedback_routes_force_the_button_rating(self):
        def fake_submit_feedback(**kwargs):
            return {
                "request_id": kwargs["request_id"],
                "rating": kwargs["rating"],
                "review_status": (
                    "accepted" if kwargs["rating"] == "up" else "pending"
                ),
            }

        with patch(
            "app.service_api.feedback_store.submit_feedback",
            side_effect=fake_submit_feedback,
        ) as submit:
            helpful = asyncio.run(
                submit_helpful_feedback(
                    HelpfulFeedbackRequest(request_id="request-up")
                )
            )
            unhelpful = asyncio.run(
                submit_unhelpful_feedback(
                    UnhelpfulFeedbackRequest(
                        request_id="request-down",
                        categories=["事实错误"],
                        comment="参数不正确",
                    )
                )
            )

        self.assertEqual(helpful["data"]["rating"], "up")
        self.assertEqual(helpful["data"]["review_status"], "accepted")
        self.assertEqual(unhelpful["data"]["rating"], "down")
        self.assertEqual(unhelpful["data"]["review_status"], "pending")
        self.assertEqual(submit.call_count, 2)

    def test_feedback_store_records_updates_stats_and_exports(self):
        temp_root = Path("D:/telebot/tmp/api-tests")
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            store = FeedbackStore(Path(directory) / "feedback.db")
            store.record_interaction(
                request_id="request-1",
                session_id="private-user-id",
                query="question",
                response="answer",
                sources=["source"],
                retrieval_backend="milvus",
                duration_seconds=1.25,
            )
            saved = store.submit_feedback(
                request_id="request-1",
                rating="down",
                categories=["回答不完整"],
                comment="missing a step",
                expected_answer="complete answer",
            )
            self.assertEqual(saved["rating"], "down")
            stats = store.stats()
            self.assertEqual(stats["interactions_total"], 1)
            self.assertEqual(stats["down_count"], 1)
            self.assertEqual(stats["category_counts"], {"回答不完整": 1})
            exported = store.export()
            self.assertEqual(exported[0]["query"], "question")
            self.assertEqual(exported[0]["categories"], ["回答不完整"])
            self.assertNotEqual(exported[0]["session_hash"], "private-user-id")

    def test_liveness_does_not_require_model_startup(self):
        response = asyncio.run(health_live())
        self.assertEqual(response["status"], "ok")

    def test_remote_readiness_does_not_require_local_milvus(self):
        remote_retriever = SimpleNamespace(active_backend="remote")
        with (
            patch.dict(os.environ, {"ENBOT_REQUIRE_MILVUS": "true"}, clear=False),
            patch(
                "app.service_api.runtime.default_retriever",
                return_value=remote_retriever,
            ),
            patch("app.service_api.runtime.initialized", True),
            patch("app.service_api.runtime.load_error", None),
            patch(
                "app.service_api._ollama_status",
                return_value={"available": True},
            ),
        ):
            response = asyncio.run(health_ready())

        self.assertEqual(response["status"], "ready")
        self.assertEqual(response["checks"]["retrieval_backend"], "remote")
        self.assertFalse(response["checks"]["milvus_required"])
        self.assertFalse(response["checks"]["milvus_active"])

    def test_optional_api_key_protects_service_metrics(self):
        with patch.dict(
            os.environ,
            {"ENBOT_EAGER_LOAD": "false", "ENBOT_API_KEY": "test-secret"},
            clear=False,
        ):
            with self.assertRaises(HTTPException) as denied:
                asyncio.run(require_api_key("wrong"))
            asyncio.run(require_api_key("test-secret"))
            allowed = asyncio.run(service_metrics())
        self.assertEqual(denied.exception.status_code, 401)
        self.assertEqual(allowed["status"], "success")

    def test_sse_event_uses_named_event_id_and_utf8_json(self):
        frame = _sse_event("delta", {"delta": "储能"}, "request-1:2")
        self.assertTrue(frame.startswith("id: request-1:2\nevent: delta\n"))
        self.assertIn('data: {"delta":"储能"}', frame)
        self.assertTrue(frame.endswith("\n\n"))

    def test_sse_chat_emits_meta_delta_sources_and_done(self):
        class FakeRequest:
            state = SimpleNamespace(request_id="stream-request")

            @staticmethod
            async def is_disconnected():
                return False

        response = ChatResponse(
            request_id="stream-request",
            data=ChatData(
                response="支持1.5倍光伏超配。",
                sources=["manual.xlsx（第 6 行）"],
                retrieval_backend="milvus",
            ),
        )

        async def collect_frames():
            with patch(
                "app.service_api._chat_impl",
                new=AsyncMock(return_value=response),
            ):
                return [
                    item
                    async for item in _chat_sse_events(
                        FakeRequest(),
                        ChatRequest(query="能否超配", session_id="sse-test"),
                    )
                ]

        frames = asyncio.run(collect_frames())
        event_names = [
            line.removeprefix("event: ")
            for frame in frames
            for line in frame.splitlines()
            if line.startswith("event: ")
        ]
        self.assertEqual(event_names[0], "meta")
        self.assertIn("delta", event_names)
        self.assertEqual(event_names[-2:], ["sources", "done"])

        payloads = [
            json.loads(line.removeprefix("data: "))
            for frame in frames
            for line in frame.splitlines()
            if line.startswith("data: ")
        ]
        deltas = "".join(item.get("delta", "") for item in payloads)
        self.assertEqual(deltas, response.data.response)
        self.assertEqual(payloads[-1]["request_id"], "stream-request")

    def test_versioned_and_compatibility_routes_are_documented(self):
        schema = app.openapi()
        for path in (
            "/v1/chat",
            "/v1/chat/stream",
            "/v1/retrieve",
            "/v1/feedback",
            "/v1/feedback/helpful",
            "/v1/feedback/unhelpful",
            "/v1/feedback/stats",
            "/v1/feedback/export",
            "/chat",
            "/retrieve",
            "/health/live",
            "/health/ready",
        ):
            self.assertIn(path, schema["paths"])


if __name__ == "__main__":
    unittest.main()
