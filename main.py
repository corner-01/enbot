"""Single entry point for the production-facing EnBot FastAPI service."""

import os

import uvicorn

from app.service_api import app


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("ENBOT_API_HOST", "127.0.0.1"),
        port=int(os.getenv("ENBOT_API_PORT", "8000")),
        workers=1,
        proxy_headers=True,
    )
