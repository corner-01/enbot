"""Production-facing FastAPI service for EnBot.

The existing :mod:`app.api` module remains the business/RAG implementation.
This module adds API lifecycle, session isolation, concurrency control, health
checks, optional authentication and versioned response schemas without changing
the retrieval behavior that passed the release evaluations.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from app.api import (
    Chatbot,
    _retrieval_debug_payload,
    get_active_model_name,
    get_llm_provider,
)
from app.feedback_store import ALLOWED_FEEDBACK_CATEGORIES, feedback_store
from memory.redis_history import RedisChatHistory
from rag.knowledge_retriever import get_knowledge_retriever


LOGGER = logging.getLogger("enbot.service")
SERVICE_NAME = "enbot"
ASSISTANT_NAME = "小英"
SERVICE_VERSION = "3.2.0"


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


def _cors_origins() -> list[str]:
    value = os.getenv("ENBOT_CORS_ORIGINS", "").strip()
    return [item.strip() for item in value.split(",") if item.strip()]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=8000)
    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.:@-]+$",
    )
    evaluate: bool = False
    ground_truth: str | None = Field(default=None, max_length=12000)
    debug: bool = False

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=8000)
    include_content: bool = False

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.:@-]+$",
    )
    rating: Literal["up", "down"]
    categories: list[str] = Field(default_factory=list, max_length=8)
    comment: str = Field(default="", max_length=4000)
    expected_answer: str = Field(default="", max_length=12000)

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, value: list[str]) -> list[str]:
        categories = list(dict.fromkeys(item.strip() for item in value if item.strip()))
        invalid = sorted(set(categories) - ALLOWED_FEEDBACK_CATEGORIES)
        if invalid:
            raise ValueError("unsupported categories: " + ", ".join(invalid))
        return categories

    @field_validator("comment", "expected_answer")
    @classmethod
    def trim_optional_text(cls, value: str) -> str:
        return value.strip()


class HelpfulFeedbackRequest(BaseModel):
    """Frontend payload for the helpful button."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_.:@-]+$",
    )


class UnhelpfulFeedbackRequest(HelpfulFeedbackRequest):
    """Frontend payload for the unhelpful button and optional correction."""

    categories: list[str] = Field(default_factory=list, max_length=8)
    comment: str = Field(default="", max_length=4000)
    expected_answer: str = Field(default="", max_length=12000)

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, value: list[str]) -> list[str]:
        categories = list(dict.fromkeys(item.strip() for item in value if item.strip()))
        invalid = sorted(set(categories) - ALLOWED_FEEDBACK_CATEGORIES)
        if invalid:
            raise ValueError("unsupported categories: " + ", ".join(invalid))
        return categories

    @field_validator("comment", "expected_answer")
    @classmethod
    def trim_optional_text(cls, value: str) -> str:
        return value.strip()


class ChatData(BaseModel):
    response: str
    sources: list[str] = Field(default_factory=list)
    retrieval_backend: str | None = None
    ragas: dict[str, Any] | None = None
    retrieval_debug: list[dict[str, Any]] | None = None


class ChatResponse(BaseModel):
    status: Literal["success"] = "success"
    request_id: str
    data: ChatData


class RetrieveData(BaseModel):
    retrieval_backend: str
    retrieval_debug: list[dict[str, Any]]


class RetrieveResponse(BaseModel):
    status: Literal["success"] = "success"
    request_id: str
    data: RetrieveData


class ServiceRuntime:
    """Own the shared retriever and bounded per-session chatbot instances."""

    def __init__(self) -> None:
        self.max_concurrency = _env_int("ENBOT_MAX_CONCURRENCY", 1)
        self.max_sessions = _env_int("ENBOT_MAX_SESSIONS", 64)
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._init_lock = asyncio.Lock()
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._sessions: OrderedDict[str, Chatbot] = OrderedDict()
        self.initialized = False
        self.load_error: str | None = None
        self.started_at = time.time()
        self.requests_total = 0
        self.requests_failed = 0
        self.total_request_seconds = 0.0

    @staticmethod
    def _build_bot(session_id: str) -> Chatbot:
        bot = Chatbot()
        if session_id != bot.session_id:
            bot.session_id = session_id
            bot.redis_memory = RedisChatHistory(session_id)
            bot._chat_history = []
        return bot

    async def start(self) -> None:
        if not _env_bool("ENBOT_EAGER_LOAD", True):
            LOGGER.info("ENBOT_EAGER_LOAD is disabled; runtime will load lazily")
            return
        try:
            await self.get_bot("default")
            LOGGER.info("EnBot runtime initialized")
        except Exception as exc:  # Keep health endpoints available for diagnosis.
            self.load_error = str(exc)
            LOGGER.exception("EnBot runtime initialization failed")

    async def get_bot(self, session_id: str) -> Chatbot:
        async with self._init_lock:
            cached = self._sessions.get(session_id)
            if cached is not None:
                self._sessions.move_to_end(session_id)
                return cached

            self.load_error = None
            try:
                bot = await run_in_threadpool(self._build_bot, session_id)
            except Exception as exc:
                self.load_error = str(exc)
                raise

            self._sessions[session_id] = bot
            self._session_locks[session_id] = asyncio.Lock()
            while len(self._sessions) > self.max_sessions:
                evicted_id, _ = self._sessions.popitem(last=False)
                self._session_locks.pop(evicted_id, None)
            self.initialized = True
            return bot

    async def chat(self, request: ChatRequest) -> dict[str, Any]:
        bot = await self.get_bot(request.session_id)
        lock = self._session_locks[request.session_id]
        started = time.perf_counter()
        self.requests_total += 1
        try:
            async with self._semaphore, lock:
                result = await run_in_threadpool(
                    bot.run,
                    request.query,
                    request.evaluate,
                    request.ground_truth,
                )
            return result
        except Exception:
            self.requests_failed += 1
            raise
        finally:
            self.total_request_seconds += time.perf_counter() - started

    async def retrieve(self, query: str) -> tuple[Any, list[Any]]:
        started = time.perf_counter()
        self.requests_total += 1
        try:
            async with self._semaphore:
                retriever = await run_in_threadpool(get_knowledge_retriever)
                if retriever is None:
                    raise RuntimeError("Knowledge retriever is unavailable")
                documents = await run_in_threadpool(retriever.invoke, query)
            return retriever, documents
        except Exception:
            self.requests_failed += 1
            raise
        finally:
            self.total_request_seconds += time.perf_counter() - started

    def default_retriever(self) -> Any | None:
        bot = self._sessions.get("default")
        return getattr(bot, "knowledge_retriever", None) if bot else None

    def metrics(self) -> dict[str, Any]:
        average = (
            self.total_request_seconds / self.requests_total
            if self.requests_total
            else 0.0
        )
        return {
            "uptime_seconds": round(time.time() - self.started_at, 3),
            "initialized": self.initialized,
            "active_sessions": len(self._sessions),
            "max_sessions": self.max_sessions,
            "max_concurrency": self.max_concurrency,
            "requests_total": self.requests_total,
            "requests_failed": self.requests_failed,
            "average_request_seconds": round(average, 4),
        }


runtime = ServiceRuntime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await runtime.start()
    yield


app = FastAPI(
    title="EnBot 家庭能源智能客服 API",
    description="Optuna QLoRA + Milvus hybrid RAG service for 小英.",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)

origins = _cors_origins()
if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", "").strip() or uuid.uuid4().hex
    request.state.request_id = request_id[:128]
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("ENBOT_API_KEY", "").strip()
    if not expected:
        return
    supplied = x_api_key or ""
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", uuid.uuid4().hex)


def _sse_event(event: str, data: dict[str, Any], event_id: str | None = None) -> str:
    """把一个事件编码为携带 UTF-8 JSON 数据的 SSE 帧。"""
    lines: list[str] = []
    if event_id:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    lines.extend(f"data: {line}" for line in encoded.splitlines() or [""])
    return "\n".join(lines) + "\n\n"


def _chunk_text(value: str, chunk_chars: int) -> list[str]:
    """把完整回答按固定字符数切分，供 SSE 分段发送。"""
    return [value[index:index + chunk_chars] for index in range(0, len(value), chunk_chars)]


def _consume_background_task_result(task: asyncio.Task[Any]) -> None:
    """客户端断开后消费后台任务结果，避免产生未处理异常。"""
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception:
        LOGGER.exception("Detached SSE chat task failed after client disconnected")


def _validate_chat_options(payload: ChatRequest) -> None:
    if payload.evaluate and not _env_bool("ENBOT_ENABLE_EVALUATION", False):
        raise HTTPException(status_code=403, detail="Online RAGAS evaluation is disabled")
    if payload.debug and not _env_bool("ENBOT_ENABLE_DEBUG", False):
        raise HTTPException(status_code=403, detail="Retrieval debug output is disabled")


def _ollama_status() -> dict[str, Any]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    expected_model = os.getenv("OLLAMA_MODEL", "enbot-v9:7b")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        response.raise_for_status()
        payload = response.json()
        names = {
            item.get("name") or item.get("model")
            for item in payload.get("models", [])
            if isinstance(item, dict)
        }
        # Ollama lists locally-created models as ``name:latest`` even when
        # OLLAMA_MODEL intentionally uses the shorter ``name`` form.
        expected_names = {expected_model}
        if ":" not in expected_model:
            expected_names.add(f"{expected_model}:latest")
        return {
            "available": bool(expected_names & names),
            "expected_model": expected_model,
            "models": sorted(name for name in names if name),
        }
    except Exception as exc:
        return {
            "available": False,
            "expected_model": expected_model,
            "error": str(exc),
        }


def _llm_status() -> dict[str, Any]:
    """Return readiness information for the configured generation provider."""
    try:
        provider = get_llm_provider()
        model = get_active_model_name()
    except ValueError as exc:
        return {"available": False, "provider": "invalid", "error": str(exc)}

    if provider == "ollama":
        return {"provider": provider, "model": model, **_ollama_status()}

    # Do not make a billable/cloud request from /health/ready. A valid key and
    # model configuration make the process ready; provider-side failures are
    # reported normally by the chat endpoint.
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = (
        os.getenv("OPENAI_BASE_URL", "").strip()
        or os.getenv("OPENAI_API_BASE", "").strip()
    )
    return {
        "available": bool(api_key),
        "provider": provider,
        "model": model,
        "base_url_configured": bool(base_url),
        **({} if api_key else {"error": "OPENAI_API_KEY is not configured"}),
    }

@app.get("/", tags=["service"])
async def root() -> dict[str, Any]:
    return {
        "service": SERVICE_NAME,
        "assistant": ASSISTANT_NAME,
        "status": "ok",
        "version": SERVICE_VERSION,
        "docs": "/docs",
        "chat": "/v1/chat",
        "chat_stream": "/v1/chat/stream",
        "retrieve": "/v1/retrieve",
        "feedback": "/v1/feedback",
    }


@app.get("/health/live", tags=["health"])
async def health_live() -> dict[str, Any]:
    return {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION}


@app.get("/health/ready", tags=["health"])
async def health_ready() -> dict[str, Any]:
    retriever = runtime.default_retriever()
    backend = getattr(retriever, "active_backend", None)
    llm = await run_in_threadpool(_llm_status)
    rag_disabled = _env_bool("ENBOT_MODEL_ONLY", False) or _env_bool("ENBOT_DISABLE_RAG", False)
    # 分布式模式下，Milvus 由远程检索服务负责；纯模型模式则不需要检索器。
    require_milvus = (
        not rag_disabled
        and _env_bool("ENBOT_REQUIRE_MILVUS", True)
        and str(backend or "").strip().lower() != "remote"
    )
    checks = {
        "runtime_initialized": runtime.initialized,
        "retriever_available": retriever is not None,
        "retrieval_backend": backend,
        "rag_enabled": not rag_disabled,
        "milvus_required": require_milvus,
        "milvus_active": backend == "milvus",
        "llm": llm,
    }
    ready = (
        runtime.initialized
        and (rag_disabled or retriever is not None)
        and (not require_milvus or backend == "milvus")
        and bool(llm.get("available"))
        and runtime.load_error is None
    )
    if not ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "checks": checks, "load_error": runtime.load_error},
        )
    return {"status": "ready", "checks": checks}


@app.get("/v1/metrics", tags=["service"], dependencies=[Depends(require_api_key)])
async def service_metrics() -> dict[str, Any]:
    return {"status": "success", "data": runtime.metrics()}


async def _chat_impl(request: Request, payload: ChatRequest) -> ChatResponse:
    _validate_chat_options(payload)
    started = time.perf_counter()
    try:
        result = await runtime.chat(payload)
    except Exception as exc:
        LOGGER.exception("Chat request failed")
        raise HTTPException(status_code=503, detail=f"EnBot runtime error: {exc}") from exc

    data: dict[str, Any] = {
        "response": result.get("output", str(result)),
        "sources": result.get("sources", []),
    }
    bot = runtime._sessions.get(payload.session_id)
    retriever = getattr(bot, "knowledge_retriever", None) if bot else None
    if retriever is not None:
        data["retrieval_backend"] = getattr(retriever, "active_backend", "unknown")
    if payload.evaluate:
        data["ragas"] = result.get("ragas")
    if payload.debug and retriever is not None:
        try:
            documents = await run_in_threadpool(retriever.invoke, payload.query)
            data["retrieval_debug"] = _retrieval_debug_payload(documents)
        except Exception as exc:
            LOGGER.warning("Retrieval debug failed: %s", exc)
            data["retrieval_debug"] = []
    request_id = _request_id(request)
    response = ChatResponse(request_id=request_id, data=ChatData(**data))
    if _env_bool("ENBOT_FEEDBACK_ENABLED", True):
        try:
            await run_in_threadpool(
                feedback_store.record_interaction,
                request_id=request_id,
                session_id=payload.session_id,
                query=payload.query,
                response=response.data.response,
                sources=response.data.sources,
                retrieval_backend=response.data.retrieval_backend,
                duration_seconds=time.perf_counter() - started,
            )
        except Exception:
            # Feedback storage must never make an otherwise valid answer fail.
            LOGGER.exception("Failed to persist interaction for feedback")
    return response


async def _chat_sse_events(
    request: Request,
    payload: ChatRequest,
) -> AsyncIterator[str]:
    """按照稳定的 SSE 事件协议，流式输出一次聊天响应。"""
    request_id = _request_id(request)
    started = time.perf_counter()
    sequence = 0

    def frame(event: str, data: dict[str, Any]) -> str:
        nonlocal sequence
        # 每个事件都使用“请求编号:事件序号”作为唯一事件 ID。
        event_id = f"{request_id}:{sequence}"
        sequence += 1
        return _sse_event(event, data, event_id)

    # 首帧先返回请求、会话和模型信息，调用方可立即确认连接成功。
    yield frame(
        "meta",
        {
            "request_id": request_id,
            "session_id": payload.session_id,
            "model": get_active_model_name(),
            "protocol_version": "1.0",
        },
    )

    # 原有 RAG 调用是阻塞式完整回答，因此放入独立异步任务中执行。
    task = asyncio.create_task(_chat_impl(request, payload))
    heartbeat_seconds = _env_int("ENBOT_SSE_HEARTBEAT_SECONDS", 10)
    response: ChatResponse | None = None
    try:
        while response is None:
            if await request.is_disconnected():
                # run_in_threadpool 无法可靠终止正在执行的模型调用。
                # 让任务在后台自然结束并释放运行时锁。
                task.add_done_callback(_consume_background_task_result)
                return
            try:
                response = await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=heartbeat_seconds,
                )
            except asyncio.TimeoutError:
                # 检索或生成尚未完成时发送心跳，避免代理和浏览器误判超时。
                yield frame(
                    "heartbeat",
                    {
                        "request_id": request_id,
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                    },
                )
    except HTTPException as exc:
        yield frame(
            "error",
            {
                "request_id": request_id,
                "code": "chat_failed",
                "status_code": exc.status_code,
                "message": exc.detail,
            },
        )
        return
    except Exception as exc:
        LOGGER.exception("SSE chat request failed")
        yield frame(
            "error",
            {
                "request_id": request_id,
                "code": "internal_error",
                "status_code": 503,
                "message": str(exc),
            },
        )
        return

    # 回答完成后按字符数切片；客户端收到 delta 后按顺序拼接。
    chunk_chars = _env_int("ENBOT_SSE_CHUNK_CHARS", 24)
    for index, chunk in enumerate(_chunk_text(response.data.response, chunk_chars)):
        if await request.is_disconnected():
            return
        yield frame(
            "delta",
            {
                "request_id": request_id,
                "index": index,
                "delta": chunk,
            },
        )
        await asyncio.sleep(0)

    # 答案发送完毕后单独返回知识来源和检索后端。
    yield frame(
        "sources",
        {
            "request_id": request_id,
            "sources": response.data.sources,
            "retrieval_backend": response.data.retrieval_backend,
        },
    )
    # done 是成功结束标记；失败时上方会发送 error 并结束生成器。
    yield frame(
        "done",
        {
            "request_id": request_id,
            "status": "success",
            "finish_reason": "stop",
            "duration_seconds": round(time.perf_counter() - started, 3),
        },
    )


@app.post(
    "/v1/chat",
    response_model=ChatResponse,
    response_model_exclude_none=True,
    tags=["chat"],
    dependencies=[Depends(require_api_key)],
)
async def chat_v1(request: Request, payload: ChatRequest) -> ChatResponse:
    return await _chat_impl(request, payload)


@app.post(
    "/v1/chat/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "SSE 事件流：meta、heartbeat、delta、sources、done/error",
            "content": {"text/event-stream": {}},
        }
    },
    tags=["chat"],
    dependencies=[Depends(require_api_key)],
)
async def chat_stream_v1(request: Request, payload: ChatRequest) -> StreamingResponse:
    _validate_chat_options(payload)
    return StreamingResponse(
        _chat_sse_events(request, payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    response_model_exclude_none=True,
    tags=["compatibility"],
    dependencies=[Depends(require_api_key)],
)
async def chat_compatibility(request: Request, payload: ChatRequest) -> ChatResponse:
    """Backward-compatible alias used by Postman and the Feishu integration."""
    return await _chat_impl(request, payload)


async def _retrieve_impl(request: Request, payload: RetrieveRequest) -> RetrieveResponse:
    if payload.include_content and not _env_bool("ENBOT_ENABLE_DEBUG", False):
        raise HTTPException(status_code=403, detail="Retrieval content output is disabled")
    try:
        retriever, documents = await runtime.retrieve(payload.query)
    except Exception as exc:
        LOGGER.exception("Retrieval request failed")
        raise HTTPException(status_code=503, detail=f"Retriever error: {exc}") from exc
    debug_payload = _retrieval_debug_payload(documents)
    if payload.include_content:
        for item, document in zip(debug_payload, documents):
            item["content"] = document.page_content
    return RetrieveResponse(
        request_id=_request_id(request),
        data=RetrieveData(
            retrieval_backend=getattr(retriever, "active_backend", "unknown"),
            retrieval_debug=debug_payload,
        ),
    )


@app.post(
    "/v1/retrieve",
    response_model=RetrieveResponse,
    tags=["retrieval"],
    dependencies=[Depends(require_api_key)],
)
async def retrieve_v1(request: Request, payload: RetrieveRequest) -> RetrieveResponse:
    return await _retrieve_impl(request, payload)


@app.post(
    "/retrieve",
    response_model=RetrieveResponse,
    tags=["compatibility"],
    dependencies=[Depends(require_api_key)],
)
async def retrieve_compatibility(request: Request, payload: RetrieveRequest) -> RetrieveResponse:
    return await _retrieve_impl(request, payload)


@app.post(
    "/v1/feedback",
    tags=["feedback"],
    dependencies=[Depends(require_api_key)],
)
async def submit_feedback(payload: FeedbackRequest) -> dict[str, Any]:
    return await _submit_feedback_record(
        request_id=payload.request_id,
        rating=payload.rating,
        categories=payload.categories,
        comment=payload.comment,
        expected_answer=payload.expected_answer,
    )


async def _submit_feedback_record(
    *,
    request_id: str,
    rating: Literal["up", "down"],
    categories: list[str],
    comment: str,
    expected_answer: str,
) -> dict[str, Any]:
    """Persist feedback through one shared validation and error boundary."""
    if not _env_bool("ENBOT_FEEDBACK_ENABLED", True):
        raise HTTPException(status_code=503, detail="Feedback collection is disabled")
    try:
        record = await run_in_threadpool(
            feedback_store.submit_feedback,
            request_id=request_id,
            rating=rating,
            categories=categories,
            comment=comment,
            expected_answer=expected_answer,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Unknown request_id; submit feedback for a recorded chat response",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"status": "success", "data": record}


@app.post(
    "/v1/feedback/helpful",
    tags=["feedback"],
    dependencies=[Depends(require_api_key)],
)
async def submit_helpful_feedback(
    payload: HelpfulFeedbackRequest,
) -> dict[str, Any]:
    """Record a frontend thumbs-up action for one completed chat request."""
    return await _submit_feedback_record(
        request_id=payload.request_id,
        rating="up",
        categories=[],
        comment="",
        expected_answer="",
    )


@app.post(
    "/v1/feedback/unhelpful",
    tags=["feedback"],
    dependencies=[Depends(require_api_key)],
)
async def submit_unhelpful_feedback(
    payload: UnhelpfulFeedbackRequest,
) -> dict[str, Any]:
    """Record a frontend thumbs-down action and optional correction details."""
    return await _submit_feedback_record(
        request_id=payload.request_id,
        rating="down",
        categories=payload.categories,
        comment=payload.comment,
        expected_answer=payload.expected_answer,
    )


@app.get(
    "/v1/feedback/stats",
    tags=["feedback"],
    dependencies=[Depends(require_api_key)],
)
async def feedback_stats() -> dict[str, Any]:
    data = await run_in_threadpool(feedback_store.stats)
    return {"status": "success", "data": data}


@app.get(
    "/v1/feedback/export",
    tags=["feedback"],
    dependencies=[Depends(require_api_key)],
)
async def export_feedback(
    limit: int = Query(default=1000, ge=1, le=10000),
) -> dict[str, Any]:
    records = await run_in_threadpool(feedback_store.export, limit=limit)
    return {
        "status": "success",
        "data": {"count": len(records), "records": records},
    }
