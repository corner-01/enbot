"""Milvus vector-store integration for EnBot RAG."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


@dataclass(frozen=True)
class VectorStoreSettings:
    backend: str
    milvus_uri: str
    milvus_token: str
    milvus_db_name: str
    milvus_collection: str
    milvus_consistency_level: str
    insert_batch_size: int

    @classmethod
    def from_env(cls) -> "VectorStoreSettings":
        backend = os.getenv("VECTOR_STORE", "milvus").strip().lower()
        if backend != "milvus":
            raise ValueError("VECTOR_STORE must be 'milvus'")
        return cls(
            backend=backend,
            milvus_uri=os.getenv(
                "MILVUS_URI", "http://127.0.0.1:19530"
            ).strip(),
            milvus_token=os.getenv("MILVUS_TOKEN", "").strip(),
            milvus_db_name=os.getenv("MILVUS_DB_NAME", "default").strip(),
            milvus_collection=os.getenv(
                "MILVUS_COLLECTION", "enbot_knowledge_v1"
            ).strip(),
            milvus_consistency_level=os.getenv(
                "MILVUS_CONSISTENCY_LEVEL", "Session"
            ).strip(),
            insert_batch_size=max(1, _env_int("MILVUS_INSERT_BATCH_SIZE", 32)),
        )


def _milvus_connection_args(settings: VectorStoreSettings) -> dict[str, Any]:
    args: dict[str, Any] = {"uri": settings.milvus_uri}
    if settings.milvus_token:
        args["token"] = settings.milvus_token
    if settings.milvus_db_name:
        args["db_name"] = settings.milvus_db_name
    return args


def _milvus_store_kwargs(settings: VectorStoreSettings) -> dict[str, Any]:
    return {
        "collection_name": settings.milvus_collection,
        "connection_args": _milvus_connection_args(settings),
        "consistency_level": settings.milvus_consistency_level,
        "auto_id": True,
        "enable_dynamic_field": True,
        "index_params": {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        },
        "search_params": {
            "metric_type": "COSINE",
            "params": {"ef": 64},
        },
    }


def _create_or_load_milvus(
    settings: VectorStoreSettings,
    embeddings: Embeddings,
    documents: Sequence[Document],
    force_rebuild: bool,
) -> tuple[Any, bool]:
    try:
        from langchain_milvus import Milvus
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise RuntimeError(
            "Milvus dependencies are missing. Run: "
            "python -m pip install -r requirements.txt"
        ) from exc

    connection_args = _milvus_connection_args(settings)
    try:
        client = MilvusClient(**connection_args)
        exists = bool(client.has_collection(settings.milvus_collection))
    except Exception as exc:
        raise RuntimeError(
            f"Cannot connect to Milvus at {settings.milvus_uri}. "
            "Start Milvus first, then rebuild the collection."
        ) from exc

    if force_rebuild and exists:
        client.drop_collection(settings.milvus_collection)
        exists = False

    kwargs = _milvus_store_kwargs(settings)
    vectorstore = Milvus(embedding_function=embeddings, **kwargs)
    if exists:
        print(
            "[OK] Loaded Milvus collection "
            f"'{settings.milvus_collection}' from {settings.milvus_uri}"
        )
        return vectorstore, False

    if not documents:
        raise RuntimeError(
            "Milvus collection does not exist and no source documents were loaded"
        )

    total = len(documents)
    for start in range(0, total, settings.insert_batch_size):
        batch = list(documents[start : start + settings.insert_batch_size])
        vectorstore.add_documents(batch)
        print(
            "[INDEX] Milvus inserted "
            f"{min(start + len(batch), total)}/{total} chunks"
        )
    print(
        "[OK] Built Milvus collection "
        f"'{settings.milvus_collection}' with {total} chunks"
    )
    return vectorstore, True


def create_or_load_vector_store(
    settings: VectorStoreSettings,
    embeddings: Embeddings,
    documents: Sequence[Document],
    force_rebuild: bool = False,
) -> tuple[Any, str, bool]:
    """Return ``(vectorstore, active_backend, created)`` for Milvus."""

    vectorstore, created = _create_or_load_milvus(
        settings, embeddings, documents, force_rebuild
    )
    return vectorstore, settings.backend, created
