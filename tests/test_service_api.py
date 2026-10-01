import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from app.service_api import (
    ChatRequest,
    FeedbackRequest,
    RetrieveRequest,
    app,
    health_live,
    require_api_key,
    service_metrics,
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

    def test_versioned_and_compatibility_routes_are_documented(self):
        schema = app.openapi()
        for path in (
            "/v1/chat",
            "/v1/retrieve",
            "/v1/feedback",
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
