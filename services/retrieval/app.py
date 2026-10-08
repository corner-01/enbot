"""Dedicated FastAPI service for embedding, hybrid search and GPU reranking."""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from rag.knowledge_retriever import get_local_knowledge_retriever


LOGGER = logging.getLogger("enbot.retrieval")
SERVICE_NAME = "enbot-retrieval"
SERVICE_VERSION = "1.0.0"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=8000)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class RetrievedDocument(BaseModel):
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieveData(BaseModel):
    retrieval_backend: str
    documents: list[RetrievedDocument]
    elapsed_seconds: float


class RetrieveResponse(BaseModel):
    status: Literal["success"] = "success"
    request_id: str
    data: RetrieveData


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = (
        os.getenv("RETRIEVAL_API_KEY", "").strip()
        or os.getenv("ENBOT_API_KEY", "").strip()
    )
    if not expected:
        return
    if not hmac.compare_digest(x_api_key or "", expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


class RetrievalRuntime:
    def __init__(self) -> None:
        self._init_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(
            _env_int("RETRIEVAL_MAX_CONCURRENCY", 1)
        )
        self.retriever: Any | None = None
        self.load_error: str | None = None
        self.started_at = time.time()
        self.requests_total = 0
        self.requests_failed = 0
        self.total_request_seconds = 0.0

    async def ensure_ready(self) -> Any:
        if self.retriever is not None:
            return self.retriever
        async with self._init_lock:
            if self.retriever is not None:
                return self.retriever
            self.load_error = None
            try:
                self.retriever = await run_in_threadpool(
                    get_local_knowledge_retriever,
                    False,
                    True,
                )
                if self.retriever is None:
                    raise RuntimeError("Local knowledge retriever is unavailable")
                if _env_bool("RETRIEVAL_EAGER_WARMUP", True):
                    warmup_query = os.getenv(
                        "RETRIEVAL_WARMUP_QUERY",
                        "储能系统故障处理",
                    ).strip()
                    if warmup_query:
                        await run_in_threadpool(
                            self.retriever.invoke,
                            warmup_query,
                        )
                return self.retriever
            except Exception as exc:
                self.load_error = str(exc)
                self.retriever = None
                LOGGER.exception("Retrieval runtime initialization failed")
                raise

    async def start(self) -> None:
        if not _env_bool("RETRIEVAL_EAGER_LOAD", True):
            return
        try:
            await self.ensure_ready()
        except Exception:
            return

    async def search(self, query: str) -> tuple[Any, list[Any], float]:
        started = time.perf_counter()
        self.requests_total += 1
        try:
            retriever = await self.ensure_ready()
            async with self._semaphore:
                documents = await run_in_threadpool(retriever.invoke, query)
            return retriever, list(documents), time.perf_counter() - started
        except Exception:
            self.requests_failed += 1
            raise
        finally:
            self.total_request_seconds += time.perf_counter() - started

    def metrics(self) -> dict[str, Any]:
        average = (
            self.total_request_seconds / self.requests_total
            if self.requests_total
            else 0.0
        )
        return {
            "uptime_seconds": round(time.time() - self.started_at, 3),
            "ready": self.retriever is not None and self.load_error is None,
            "requests_total": self.requests_total,
            "requests_failed": self.requests_failed,
            "average_request_seconds": round(average, 4),
            "max_concurrency": _env_int("RETRIEVAL_MAX_CONCURRENCY", 1),
        }


runtime = RetrievalRuntime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await runtime.start()
    yield


app = FastAPI(
    title="EnBot GPU Retrieval API",
    description="Qwen embedding, Milvus and BM25 hybrid search, and GPU reranking",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", "").strip() or uuid.uuid4().hex
    request.state.request_id = request_id[:128]
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", uuid.uuid4().hex)


def _gpu_status() -> dict[str, Any]:
    available = torch.cuda.is_available()
    details: dict[str, Any] = {
        "available": available,
        "embedding_device": os.getenv("EMBEDDING_DEVICE", "cuda"),
        "reranker_device": os.getenv("RERANKER_DEVICE", "cuda"),
    }
    if available:
        details.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "allocated_mib": round(torch.cuda.memory_allocated(0) / 1048576, 1),
                "reserved_mib": round(torch.cuda.memory_reserved(0) / 1048576, 1),
            }
        )
    return details


@app.get("/", tags=["service"])
async def service_info() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "retrieve": "/v1/retrieve",
    }


@app.get("/health/live", tags=["health"])
async def health_live() -> dict[str, Any]:
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/health/ready", tags=["health"])
async def health_ready() -> dict[str, Any]:
    gpu = _gpu_status()
    expected_gpu = {
        os.getenv("EMBEDDING_DEVICE", "cuda").casefold(),
        os.getenv("RERANKER_DEVICE", "cuda").casefold(),
    }
    gpu_required = any(device.startswith("cuda") for device in expected_gpu)
    ready = (
        runtime.retriever is not None
        and runtime.load_error is None
        and (gpu["available"] or not gpu_required)
    )
    if not ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "load_error": runtime.load_error,
                "gpu": gpu,
            },
        )
    return {"status": "ready", "gpu": gpu}


@app.get(
    "/v1/metrics",
    tags=["service"],
    dependencies=[Depends(require_api_key)],
)
async def service_metrics() -> dict[str, Any]:
    return {"status": "success", "data": runtime.metrics()}


@app.post(
    "/v1/retrieve",
    response_model=RetrieveResponse,
    tags=["retrieval"],
    dependencies=[Depends(require_api_key)],
)
async def retrieve(request: Request, payload: RetrieveRequest) -> RetrieveResponse:
    try:
        retriever, documents, elapsed = await runtime.search(payload.query)
    except Exception as exc:
        LOGGER.exception("Retrieval request failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Retriever error: {exc}",
        ) from exc
    return RetrieveResponse(
        request_id=_request_id(request),
        data=RetrieveData(
            retrieval_backend=getattr(retriever, "active_backend", "unknown"),
            documents=[
                RetrievedDocument(
                    content=document.page_content,
                    metadata=dict(document.metadata or {}),
                )
                for document in documents
            ],
            elapsed_seconds=round(elapsed, 4),
        ),
    )

