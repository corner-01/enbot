"""EnBot MCP adapter.

This process does not load the model, Milvus, or the embedding/reranking
models. It exposes a small set of MCP tools and delegates every operation to
the existing production FastAPI service.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Literal

import requests
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError


SERVER_NAME = "enbot-mcp"
DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8020


def _env_float(name: str, default: float, minimum: float = 0.1) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


class EnBotApiError(ToolError):
    """Raised when the existing EnBot HTTP API cannot complete a tool call."""


class EnBotApiClient:
    """Small, stateless client for the production EnBot API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("ENBOT_MCP_API_BASE_URL") or DEFAULT_API_BASE_URL
        ).rstrip("/")
        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv("ENBOT_MCP_API_KEY", os.getenv("ENBOT_API_KEY", ""))
        )
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else _env_float("ENBOT_MCP_TIMEOUT_SECONDS", 300.0)
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        try:
            response = requests.request(
                method,
                f"{self.base_url}{path}",
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise EnBotApiError(
                f"无法连接 EnBot API（{self.base_url}）：{exc}"
            ) from exc

        try:
            body = response.json()
        except ValueError:
            body = {"detail": response.text.strip() or "empty response"}

        if not response.ok:
            detail = body.get("detail", body) if isinstance(body, dict) else body
            raise EnBotApiError(
                f"EnBot API 请求失败（HTTP {response.status_code}）：{detail}"
            )
        if not isinstance(body, dict):
            raise EnBotApiError("EnBot API 返回了非对象 JSON")
        return body

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health/ready")

    def ask(
        self,
        query: str,
        session_id: str = "mcp-default",
        debug: bool = False,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/chat",
            {"query": query, "session_id": session_id, "debug": debug},
        )

    def retrieve(self, query: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/retrieve",
            {"query": query, "include_content": False},
        )

    def feedback(
        self,
        request_id: str,
        rating: Literal["up", "down"],
        categories: list[str] | None = None,
        comment: str = "",
        expected_answer: str = "",
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/feedback",
            {
                "request_id": request_id,
                "rating": rating,
                "categories": categories or [],
                "comment": comment,
                "expected_answer": expected_answer,
            },
        )


api_client = EnBotApiClient()
mcp = MCPServer(SERVER_NAME)


@mcp.tool()
def enbot_health() -> dict[str, Any]:
    """检查 EnBot、Milvus 检索后端和 Ollama 模型是否已经就绪。"""

    return api_client.health()


@mcp.tool()
def enbot_ask(
    query: str,
    session_id: str = "mcp-default",
    debug: bool = False,
) -> dict[str, Any]:
    """向家庭储能助手提问，返回回答、知识来源和请求编号。

    Args:
        query: 家庭储能、逆变器、电池、告警或操作方面的问题。
        session_id: 会话标识；同一轮连续问答应保持一致。
        debug: 是否请求检索调试信息；服务端未开启调试时请保持 false。
    """

    return api_client.ask(query=query, session_id=session_id, debug=debug)


@mcp.tool()
def enbot_retrieve(query: str) -> dict[str, Any]:
    """只检索家庭储能知识库，返回命中文档的来源和相关度信息。"""

    return api_client.retrieve(query=query)


@mcp.tool()
def enbot_feedback(
    request_id: str,
    rating: Literal["up", "down"],
    categories: list[str] | None = None,
    comment: str = "",
    expected_answer: str = "",
) -> dict[str, Any]:
    """对一次 EnBot 回答提交反馈。

    Args:
        request_id: enbot_ask 返回的 request_id。
        rating: up 表示有帮助，down 表示没有帮助。
        categories: 没有帮助时的问题分类；必须使用服务端允许的中文分类。
        comment: 对问题的补充说明。
        expected_answer: 建议的正确答案，用于后续人工审核和数据改进。
    """

    return api_client.feedback(
        request_id=request_id,
        rating=rating,
        categories=categories,
        comment=comment,
        expected_answer=expected_answer,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the EnBot MCP adapter")
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default=os.getenv("ENBOT_MCP_TRANSPORT", "stdio"),
        help="stdio for a local MCP host, streamable-http for a network endpoint",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("ENBOT_MCP_HOST", DEFAULT_HTTP_HOST),
        help="HTTP bind address; keep 127.0.0.1 unless access is protected",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("ENBOT_MCP_PORT", str(DEFAULT_HTTP_PORT))),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.transport == "stdio":
        mcp.run(transport="stdio")
        return
    mcp.run(
        transport="streamable-http",
        host=args.host,
        port=args.port,
        stateless_http=True,
        json_response=True,
    )


if __name__ == "__main__":
    main()
