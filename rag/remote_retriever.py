"""HTTP client that exposes the retrieval GPU node as a LangChain retriever."""

from __future__ import annotations

import os
import uuid
from typing import Any, Mapping, Sequence

import requests
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float, minimum: float = 0.1) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _validated_api_key(value: str) -> str:
    """Return a header-safe API key or raise a configuration error."""
    value = value.strip()
    if value and not value.isascii():
        raise ValueError(
            "RETRIEVAL_API_KEY must contain ASCII characters only; "
            "replace any Chinese placeholder text with the real key"
        )
    return value


class RemoteRetrievalError(RuntimeError):
    """Raised when the dedicated retrieval node cannot serve a request."""


class RemoteKnowledgeRetriever(BaseRetriever):
    """Call a remote hybrid retriever without loading local embedding models."""

    api_url: str
    api_key: str = ""
    timeout_seconds: float = 8.0
    local_fallback: bool = False
    active_backend: str = "remote"
    embeddings: Any = None
    vectorstore: Any = None
    hybrid_engine: Any = None
    _session: requests.Session = PrivateAttr(default_factory=requests.Session)

    @classmethod
    def from_env(cls) -> "RemoteKnowledgeRetriever":
        api_url = os.getenv("RETRIEVAL_API_URL", "").strip().rstrip("/")
        if not api_url:
            raise ValueError(
                "RETRIEVAL_API_URL is required when RETRIEVAL_MODE=remote"
            )
        return cls(
            api_url=api_url,
            api_key=_validated_api_key(os.getenv("RETRIEVAL_API_KEY", "")),
            timeout_seconds=_env_float("RETRIEVAL_TIMEOUT_SECONDS", 8.0),
            local_fallback=_env_bool("RETRIEVAL_LOCAL_FALLBACK", False),
        )

    @property
    def endpoint(self) -> str:
        return f"{self.api_url}/v1/retrieve"

    def _fallback_or_raise(self, query: str, exc: Exception) -> list[Document]:
        if self.local_fallback:
            from rag.knowledge_retriever import get_local_knowledge_retriever

            local = get_local_knowledge_retriever(raise_on_error=True)
            if local is not None:
                return list(local.invoke(query))
        raise RemoteRetrievalError(str(exc)) from exc

    @staticmethod
    def _documents_from_payload(payload: Mapping[str, Any]) -> list[Document]:
        data = payload.get("data")
        if not isinstance(data, Mapping):
            raise RemoteRetrievalError("Remote retrieval response is missing data")

        raw_documents = data.get("documents")
        if not isinstance(raw_documents, Sequence) or isinstance(
            raw_documents, (str, bytes)
        ):
            raw_documents = data.get("retrieval_debug")
        if not isinstance(raw_documents, Sequence) or isinstance(
            raw_documents, (str, bytes)
        ):
            raise RemoteRetrievalError(
                "Remote retrieval response is missing documents"
            )

        backend = str(data.get("retrieval_backend") or "remote")
        documents: list[Document] = []
        for rank, item in enumerate(raw_documents, start=1):
            if not isinstance(item, Mapping):
                continue
            content = item.get("content") or item.get("page_content")
            if not isinstance(content, str) or not content.strip():
                continue
            raw_metadata = item.get("metadata")
            if isinstance(raw_metadata, Mapping):
                metadata = dict(raw_metadata)
            else:
                metadata = {
                    str(key): value
                    for key, value in item.items()
                    if key not in {"content", "page_content", "preview", "metadata"}
                }
            metadata.setdefault("remote_rank", rank)
            metadata.setdefault("retrieval_backend", backend)
            documents.append(
                Document(page_content=content.strip(), metadata=metadata)
            )
        return documents

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        try:
            api_key = _validated_api_key(self.api_key)
        except ValueError as exc:
            raise RemoteRetrievalError(str(exc)) from exc
        headers = {
            "Content-Type": "application/json",
            "X-Request-ID": uuid.uuid4().hex,
        }
        if api_key:
            headers["X-API-Key"] = api_key
        try:
            response = self._session.post(
                self.endpoint,
                headers=headers,
                json={"query": query},
                timeout=self.timeout_seconds,
            )
            if response.status_code >= 400:
                preview = response.text.strip().replace("\n", " ")[:500]
                raise RemoteRetrievalError(
                    f"Remote retrieval returned HTTP {response.status_code}: {preview}"
                )
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise RemoteRetrievalError(
                    "Remote retrieval returned a non-object JSON response"
                )
            if payload.get("status") != "success":
                raise RemoteRetrievalError(
                    str(payload.get("error") or "Remote retrieval failed")
                )
            return self._documents_from_payload(payload)
        except (
            requests.RequestException,
            ValueError,
            TypeError,
            RemoteRetrievalError,
        ) as exc:
            return self._fallback_or_raise(query, exc)
