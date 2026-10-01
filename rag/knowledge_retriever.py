"""Knowledge loading, vector indexing, and hybrid retrieval for EnBot."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from rag.hybrid_search import HybridSearchEngine
from rag.vector_store import VectorStoreSettings, create_or_load_vector_store


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_knowledge_retriever_cache: Optional[BaseRetriever] = None


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    return float(value) if value else default


class LocalSentenceEmbeddings(Embeddings):
    """Thread-safe wrapper around the project's local embedding model."""

    def __init__(self, model_path: str):
        # RAGAS reads ``embeddings.model`` for telemetry metadata and expects a
        # string rather than a SentenceTransformer instance.
        self.model = str(model_path)
        self.device = os.getenv(
            "EMBEDDING_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        self._encoder = SentenceTransformer(model_path, device=self.device)
        self._encode_lock = threading.Lock()

    def embed_documents(self, texts):
        with self._encode_lock:
            if hasattr(self._encoder, "encode_document"):
                embeddings = self._encoder.encode_document(
                    texts,
                    normalize_embeddings=True,
                    batch_size=_env_int("EMBEDDING_BATCH_SIZE", 4),
                )
            else:
                embeddings = self._encoder.encode(
                    texts,
                    prompt_name="document",
                    normalize_embeddings=True,
                    batch_size=_env_int("EMBEDDING_BATCH_SIZE", 4),
                )
        return embeddings.tolist()

    def embed_query(self, text):
        with self._encode_lock:
            if hasattr(self._encoder, "encode_query"):
                embeddings = self._encoder.encode_query(
                    [text], normalize_embeddings=True
                )
            else:
                embeddings = self._encoder.encode(
                    [text],
                    prompt_name="query",
                    normalize_embeddings=True,
                )
        return embeddings[0].tolist()


class KnowledgeRetriever(BaseRetriever):
    """Expose hybrid retrieval through LangChain's retriever interface."""

    base_retriever: BaseRetriever
    embeddings: Embeddings
    vectorstore: Any
    hybrid_engine: Optional[Any] = None
    active_backend: str = ""

    def _get_relevant_documents(self, query: str, *, run_manager):
        if self.hybrid_engine is not None:
            return self.hybrid_engine.search(query)
        return self.base_retriever.invoke(query)


def _safe_metadata_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _sanitize_document(document: Document) -> Document:
    metadata = {
        str(key): _safe_metadata_value(value)
        for key, value in (document.metadata or {}).items()
        if value is not None
    }
    return Document(page_content=document.page_content, metadata=metadata)


def _record_text(value: Any) -> str:
    """Convert a JSON Q&A field into stable searchable text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, default=str).strip()


def _record_id(source: str, row: Optional[int] = None) -> str:
    """Build a stable parent-record id shared by all chunks of one record."""
    return f"{source}::row:{row}" if row is not None else f"{source}::document"


def _json_qa_document(
    record: Any,
    *,
    source: str,
    row_index: int,
) -> Optional[Document]:
    """Convert one Alpaca-style instruction/input/output row to a document."""
    if not isinstance(record, dict):
        return None

    instruction = _record_text(record.get("instruction"))
    input_text = _record_text(record.get("input"))
    output = _record_text(record.get("output"))
    if not output or not (instruction or input_text):
        return None

    question_parts = [part for part in (instruction, input_text) if part]
    question = "\n".join(question_parts)
    page_content = f"Question: {question}\nAnswer: {output}"
    return Document(
        page_content=page_content,
        metadata={
            "source": source,
            "row": row_index,
            "format": "instruction_input_output",
            "record_id": _record_id(source, row_index),
        },
    )


def _load_knowledge_corpus(
    folder_path: str | Path,
) -> tuple[list[Document], dict[str, Document]]:
    """Load source records and return searchable chunks plus full parents."""
    root_path = Path(folder_path).expanduser()
    if not root_path.exists():
        print(f"[WARN] Knowledge source path does not exist: {root_path}")
        return [], {}

    raw_documents: list[Document] = []
    parent_documents: dict[str, Document] = {}
    seen_content: set[str] = set()
    duplicate_count = 0
    invalid_record_count = 0
    file_count = 0

    def append_unique(document: Optional[Document]) -> None:
        nonlocal duplicate_count, invalid_record_count
        if document is None:
            invalid_record_count += 1
            return
        content = document.page_content.strip()
        if len(content) < 10:
            invalid_record_count += 1
            return
        record_id = str((document.metadata or {}).get("record_id", "")).strip()
        if record_id:
            # Keep aliases for duplicate sources. An older vector index may still
            # return the duplicate filename even though only one copy is indexed.
            parent_documents.setdefault(record_id, _sanitize_document(document))
        if content in seen_content:
            duplicate_count += 1
            return
        seen_content.add(content)
        raw_documents.append(document)

    for file_path in sorted(root_path.rglob("*")):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in {".txt", ".xlsx", ".json"}:
            continue
        file_count += 1
        source = str(file_path.relative_to(root_path))
        try:
            if suffix == ".txt":
                text = file_path.read_text(encoding="utf-8").strip()
                if text:
                    append_unique(
                        Document(
                            page_content=text,
                            metadata={
                                "source": source,
                                "record_id": _record_id(source),
                            },
                        )
                    )
            elif suffix == ".xlsx":
                dataframe = pd.read_excel(file_path, engine="openpyxl")
                for row_index, row in dataframe.iterrows():
                    fields = [
                        f"{column}: {value}"
                        for column, value in row.items()
                        if pd.notna(value)
                    ]
                    text = "\n".join(fields).strip()
                    append_unique(
                        Document(
                            page_content=text,
                            metadata={
                                "source": source,
                                "row": int(row_index) + 2,
                                "record_id": _record_id(
                                    source, int(row_index) + 2
                                ),
                            },
                        )
                    )
            else:
                payload = json.loads(file_path.read_text(encoding="utf-8-sig"))
                records = payload if isinstance(payload, list) else [payload]
                for row_index, record in enumerate(records, start=1):
                    append_unique(
                        _json_qa_document(
                            record,
                            source=source,
                            row_index=row_index,
                        )
                    )
        except Exception as exc:
            print(f"[WARN] Failed to read {file_path}: {exc}")

    if not raw_documents:
        print(
            f"[WARN] No usable TXT/XLSX/JSON documents found under {root_path} "
            f"(files checked: {file_count})"
        )
        return [], {}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_env_int("RAG_CHUNK_SIZE", 800),
        chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 100),
        separators=["\n\n", "\n", "。", "！", "？", ";", "；", " ", ""],
        keep_separator=True,
        add_start_index=True,
    )
    chunks = [_sanitize_document(doc) for doc in splitter.split_documents(raw_documents)]
    print(
        f"[OK] Loaded {file_count} knowledge files, "
        f"{len(raw_documents)} unique records, {len(chunks)} chunks, "
        f"{duplicate_count} duplicates and {invalid_record_count} invalid records skipped"
    )
    return chunks, parent_documents


def load_knowledge_chunks(folder_path: str | Path) -> list[Document]:
    """Load TXT/XLSX/JSON sources and return dense/BM25 search chunks."""
    chunks, _ = _load_knowledge_corpus(folder_path)
    return chunks


def reset_knowledge_retriever_cache() -> None:
    global _knowledge_retriever_cache
    _knowledge_retriever_cache = None


def get_knowledge_retriever(
    force_rebuild: bool = False,
    raise_on_error: bool = False,
) -> Optional[BaseRetriever]:
    """Load or build the configured vector index and hybrid retriever."""
    global _knowledge_retriever_cache
    if _knowledge_retriever_cache is not None and not force_rebuild:
        return _knowledge_retriever_cache

    print("=" * 60)
    print("Initializing EnBot knowledge retriever")
    print("=" * 60)

    try:
        settings = VectorStoreSettings.from_env()
        folder_path = os.getenv(
            "KNOWLEDGE_FILE", str(PROJECT_ROOT / "knowledge_data")
        )
        embedding_path = os.getenv(
            "EMBEDDING_MODEL_PATH",
            str(PROJECT_ROOT / "models" / "Qwen3-8B-native"),
        )

        print(f"[INFO] Loading local embedding model: {embedding_path}")
        embeddings = LocalSentenceEmbeddings(model_path=embedding_path)
        chunks, parent_documents = _load_knowledge_corpus(folder_path)

        vectorstore, active_backend, created = create_or_load_vector_store(
            settings=settings,
            embeddings=embeddings,
            documents=chunks,
            force_rebuild=force_rebuild,
        )

        dense_k = _env_int("RAG_DENSE_K", 30)
        base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": dense_k},
        )
        reranker_path = os.getenv(
            "RERANKER_MODEL_PATH",
            str(PROJECT_ROOT / "models" / "bge-reranker-v2-m3"),
        )
        hybrid_engine = HybridSearchEngine(
            vectorstore=vectorstore,
            dense_retriever=base_retriever,
            documents=chunks,
            parent_documents=parent_documents,
            reranker_path=reranker_path,
            dense_k=dense_k,
            lexical_k=_env_int("RAG_LEXICAL_K", 50),
            rerank_k=_env_int("RAG_RERANK_K", 10),
            final_k=_env_int("RAG_FINAL_K", 2),
            rrf_k=_env_float("RAG_RRF_K", 60.0),
            dense_weight=_env_float("RAG_DENSE_WEIGHT", 1.0),
            lexical_weight=_env_float("RAG_LEXICAL_WEIGHT", 1.0),
            rerank_relative_threshold=_env_float(
                "RAG_RERANK_RELATIVE_THRESHOLD", 0.8
            ),
            rerank_absolute_threshold=_env_float(
                "RAG_RERANK_ABSOLUTE_THRESHOLD", 0.1
            ),
            rerank_min_score=_env_float("RAG_RERANK_MIN_SCORE", -1.0),
            rerank_tie_epsilon=_env_float("RAG_RERANK_TIE_EPSILON", 0.0001),
            parent_full_max_chars=_env_int("RAG_PARENT_FULL_MAX_CHARS", 12000),
        )
        retriever = KnowledgeRetriever(
            base_retriever=base_retriever,
            embeddings=embeddings,
            vectorstore=vectorstore,
            hybrid_engine=hybrid_engine,
            active_backend=active_backend,
        )
        _knowledge_retriever_cache = retriever
        action = "built" if created else "loaded"
        print(f"[OK] Knowledge retriever ready: backend={active_backend}, index={action}")
        return retriever
    except Exception as exc:
        print(f"[ERROR] Knowledge retriever initialization failed: {exc}")
        if raise_on_error:
            raise
        return None
