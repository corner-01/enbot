"""Independent FastAPI service for EnBot multimodal diagnosis.

The service accepts one image, runs the local OCR/VLM evidence pipeline, and
delegates only verified in-domain queries to the existing EnBot text API. It
does not import the text RAG runtime, so both services remain independent.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import re
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

# Keep default model artifacts on D: even when Uvicorn is launched directly
# instead of through scripts/11_start_vision_api.ps1.
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", "D:/AI-Cache/paddle")
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "BOS")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from services.vision.diagnosis_pipeline import (
    EnBotAPIResponseError,
    EnBotAPIUnavailableError,
    MultimodalDiagnosisPipeline,
)
from services.vision.image_preprocessor import ImagePreprocessingError
from services.vision.vision_client import (
    VisionModelResponseError,
    VisionServiceUnavailableError,
)


LOGGER = logging.getLogger("enbot.vision")
SERVICE_NAME = "enbot-vision"
SERVICE_VERSION = "1.0.0"
DEFAULT_UPLOAD_DIR = Path(__file__).resolve().parents[2] / "tmp" / "vision-uploads"
ALLOWED_CONTENT_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().casefold() in {"1", "true", "yes", "on"}


def _env_positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    value = int(raw_value) if raw_value else default
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _env_positive_float(name: str, default: float) -> float:
    raw_value = os.getenv(name, "").strip()
    value = float(raw_value) if raw_value else default
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return value


def _cors_origins() -> list[str]:
    raw_value = os.getenv("VISION_CORS_ORIGINS", "").strip()
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _upload_directory() -> Path:
    configured = os.getenv("VISION_UPLOAD_DIR", "").strip()
    directory = Path(configured) if configured else DEFAULT_UPLOAD_DIR
    return directory.expanduser().resolve()


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", uuid.uuid4().hex))


async def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    expected = os.getenv("VISION_API_KEY", "").strip()
    if expected and not (
        x_api_key and hmac.compare_digest(x_api_key.encode(), expected.encode())
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


class VisionServiceRuntime:
    """Own the lazy diagnosis pipeline and process-wide concurrency limit."""

    def __init__(self) -> None:
        self.max_concurrency = _env_positive_int("VISION_MAX_CONCURRENCY", 1)
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._init_lock = asyncio.Lock()
        self._pipeline: MultimodalDiagnosisPipeline | None = None
        self.started_at = time.time()
        self.requests_total = 0
        self.requests_failed = 0
        self.total_request_seconds = 0.0

    async def get_pipeline(self) -> MultimodalDiagnosisPipeline:
        if self._pipeline is not None:
            return self._pipeline
        async with self._init_lock:
            if self._pipeline is None:
                self._pipeline = await run_in_threadpool(MultimodalDiagnosisPipeline)
        return self._pipeline

    async def diagnose(
        self,
        image_path: Path,
        *,
        question: str,
        session_id: str,
    ) -> Any:
        started = time.perf_counter()
        self.requests_total += 1
        try:
            pipeline = await self.get_pipeline()
            async with self._semaphore:
                return await run_in_threadpool(
                    pipeline.run,
                    image_path,
                    question=question,
                    session_id=session_id,
                )
        except Exception:
            self.requests_failed += 1
            raise
        finally:
            self.total_request_seconds += time.perf_counter() - started

    async def close(self) -> None:
        pipeline = self._pipeline
        self._pipeline = None
        if pipeline is not None:
            await run_in_threadpool(pipeline.close)

    def metrics(self) -> dict[str, Any]:
        average = (
            self.total_request_seconds / self.requests_total
            if self.requests_total
            else 0.0
        )
        return {
            "initialized": self._pipeline is not None,
            "max_concurrency": self.max_concurrency,
            "requests_total": self.requests_total,
            "requests_failed": self.requests_failed,
            "average_request_seconds": round(average, 6),
            "uptime_seconds": round(time.time() - self.started_at, 3),
        }


runtime = VisionServiceRuntime()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    upload_dir = _upload_directory()
    upload_dir.mkdir(parents=True, exist_ok=True)
    # Starlette may spool large multipart bodies before the endpoint runs.
    # This service is isolated, so keep that process-wide temporary data on D:.
    previous_tempdir = tempfile.tempdir
    tempfile.tempdir = str(upload_dir)
    try:
        yield
    finally:
        await runtime.close()
        tempfile.tempdir = previous_tempdir


app = FastAPI(
    title="EnBot Vision API",
    description="Local OCR + vision model + grounded EnBot RAG diagnosis service",
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
    )


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    incoming = request.headers.get("X-Request-ID", "").strip()
    request.state.request_id = (
        incoming if REQUEST_ID_PATTERN.fullmatch(incoming) else uuid.uuid4().hex
    )
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "request_id": _request_id(request),
            "error": {"code": "http_error", "message": str(exc.detail)},
        },
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled vision service error", exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "request_id": _request_id(request),
            "error": {
                "code": "internal_error",
                "message": "The vision service encountered an unexpected error",
            },
        },
    )


async def _save_upload(upload: UploadFile) -> Path:
    content_type = (upload.content_type or "").casefold()
    suffix = ALLOWED_CONTENT_TYPES.get(content_type)
    if suffix is None:
        await upload.close()
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported image type. Use PNG, JPEG, WEBP, BMP, or TIFF.",
        )

    max_mib = _env_positive_float("VISION_MAX_IMAGE_MIB", 10.0)
    max_bytes = int(max_mib * 1024 * 1024)
    upload_dir = _upload_directory()
    upload_dir.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    written = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=suffix,
            prefix="upload-",
            dir=upload_dir,
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            while chunk := await upload.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=f"Image exceeds the {max_mib:g} MiB upload limit",
                    )
                destination.write(chunk)
        if written == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded image is empty",
            )
        return temporary_path
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


async def _dependency_status() -> tuple[bool, dict[str, Any]]:
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    enbot_url = os.getenv("ENBOT_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    model = os.getenv("VISION_MODEL", "qwen3-vl:2b-instruct").strip()
    details: dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
        try:
            response = await client.get(f"{ollama_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            names = {
                str(item.get("name") or item.get("model") or "")
                for item in models
                if isinstance(item, dict)
            }
            model_ready = model in names
            details["ollama"] = {"reachable": True, "model_ready": model_ready}
        except Exception as exc:
            details["ollama"] = {"reachable": False, "error": str(exc)[:200]}

        try:
            response = await client.get(f"{enbot_url}/health/live")
            response.raise_for_status()
            details["enbot_api"] = {"reachable": True}
        except Exception as exc:
            details["enbot_api"] = {"reachable": False, "error": str(exc)[:200]}

    ready = bool(
        details["ollama"].get("reachable")
        and details["ollama"].get("model_ready")
        and details["enbot_api"].get("reachable")
    )
    return ready, details


@app.get("/", tags=["service"])
async def service_info() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "docs": "/docs",
        "diagnose": "/v1/diagnose",
    }


@app.get("/health/live", tags=["health"])
async def health_live() -> dict[str, Any]:
    return {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION}


@app.get("/health/ready", tags=["health"])
async def health_ready() -> JSONResponse:
    ready, dependencies = await _dependency_status()
    return JSONResponse(
        status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "ready" if ready else "not_ready",
            "service": SERVICE_NAME,
            "dependencies": dependencies,
            "runtime": runtime.metrics(),
        },
    )


@app.get("/metrics", dependencies=[Depends(verify_api_key)], tags=["service"])
async def metrics() -> dict[str, Any]:
    return {"status": "ok", "service": SERVICE_NAME, "data": runtime.metrics()}


@app.post("/v1/diagnose", dependencies=[Depends(verify_api_key)], tags=["diagnosis"])
async def diagnose_image(
    request: Request,
    image: UploadFile = File(..., description="PNG/JPEG/WEBP/BMP/TIFF device image"),
    question: str = Form(
        default="Please analyze the device status or issue shown in this image.",
        max_length=4000,
    ),
    session_id: str = Form(default="vision-default", max_length=128),
    debug: bool = Form(default=False),
) -> JSONResponse:
    if debug and not _env_bool("VISION_ENABLE_DEBUG", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Debug output is disabled",
        )

    saved_path = await _save_upload(image)
    try:
        result = await runtime.diagnose(
            saved_path,
            question=question.strip(),
            session_id=session_id.strip(),
        )
    except (ValueError, ImagePreprocessingError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except (VisionServiceUnavailableError, EnBotAPIUnavailableError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except (VisionModelResponseError, EnBotAPIResponseError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    finally:
        saved_path.unlink(missing_ok=True)

    data = result.to_dict(include_debug=debug)
    response_status = "success" if result.status == "success" else result.status
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": response_status,
            "request_id": _request_id(request),
            "data": data,
        },
    )
