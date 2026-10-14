import asyncio
import os
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException
from langchain_core.documents import Document

from rag.knowledge_retriever import (
    get_knowledge_retriever,
    reset_knowledge_retriever_cache,
)
from rag.remote_retriever import (
    RemoteKnowledgeRetriever,
    RemoteRetrievalError,
)
from services.retrieval.app import app as retrieval_app
from services.retrieval.app import require_api_key


class RemoteRetrieverTests(unittest.TestCase):
    def tearDown(self):
        reset_knowledge_retriever_cache()

    def test_remote_response_becomes_langchain_documents(self):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "status": "success",
            "data": {
                "retrieval_backend": "milvus",
                "documents": [
                    {
                        "content": "Battery alarm handling",
                        "metadata": {
                            "source": "manual.txt",
                            "reranker_score": 0.91,
                        },
                    }
                ],
            },
        }
        retriever = RemoteKnowledgeRetriever(
            api_url="http://retrieval-node:8100",
            api_key="secret",
        )
        with patch.object(retriever._session, "post", return_value=response) as post:
            documents = retriever.invoke("alarm")

        self.assertEqual(len(documents), 1)
        self.assertIsInstance(documents[0], Document)
        self.assertEqual(documents[0].metadata["source"], "manual.txt")
        self.assertEqual(documents[0].metadata["retrieval_backend"], "milvus")
        request = post.call_args
        self.assertEqual(request.kwargs["json"], {"query": "alarm"})
        self.assertEqual(request.kwargs["headers"]["X-API-Key"], "secret")

    def test_remote_error_does_not_silently_load_local_models(self):
        response = Mock()
        response.status_code = 503
        response.text = "not ready"
        retriever = RemoteKnowledgeRetriever(
            api_url="http://retrieval-node:8100",
            local_fallback=False,
        )
        with patch.object(retriever._session, "post", return_value=response):
            with self.assertRaises(RemoteRetrievalError):
                retriever.invoke("alarm")

    def test_factory_uses_remote_mode(self):
        with patch.dict(
            os.environ,
            {
                "RETRIEVAL_MODE": "remote",
                "RETRIEVAL_API_URL": "http://retrieval-node:8100",
                "RETRIEVAL_API_KEY": "secret",
            },
            clear=False,
        ):
            reset_knowledge_retriever_cache()
            retriever = get_knowledge_retriever(raise_on_error=True)
        self.assertIsInstance(retriever, RemoteKnowledgeRetriever)

    def test_factory_rejects_non_ascii_api_key_before_http_request(self):
        with patch.dict(
            os.environ,
            {
                "RETRIEVAL_API_URL": "http://retrieval-node:8100",
                "RETRIEVAL_API_KEY": "请填写真实密钥",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "ASCII characters only"):
                RemoteKnowledgeRetriever.from_env()

    def test_retrieval_service_routes_are_documented(self):
        schema = retrieval_app.openapi()
        for path in (
            "/health/live",
            "/health/ready",
            "/v1/metrics",
            "/v1/retrieve",
        ):
            self.assertIn(path, schema["paths"])

    def test_retrieval_api_key_is_enforced(self):
        with patch.dict(os.environ, {"RETRIEVAL_API_KEY": "secret"}, clear=False):
            with self.assertRaises(HTTPException) as denied:
                asyncio.run(require_api_key("wrong"))
            asyncio.run(require_api_key("secret"))
        self.assertEqual(denied.exception.status_code, 401)


if __name__ == "__main__":
    unittest.main()
