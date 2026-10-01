"""Feishu long-connection bridge for the local EnBot API.

The process receives im.message.receive_v1 events over Feishu's official
WebSocket client, queues text messages, calls the already-running EnBot HTTP
API, and replies to the original Feishu message. It intentionally does not
import the RAG stack, so it never loads a second copy of the local models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import queue
import re
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import lark_oapi as lark
import requests
from dotenv import load_dotenv
from lark_oapi.api.im.v1 import (
    P2ImMessageReceiveV1,
    ReplyMessageRequest,
    ReplyMessageRequestBody,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

LOGGER = logging.getLogger("enbot.feishu")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _csv_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


@dataclass(frozen=True)
class FeishuConfig:
    app_id: str
    app_secret: str
    backend: str
    enbot_api_url: str
    enbot_api_key: str
    ollama_api_url: str
    ollama_model: str
    enbot_timeout_seconds: int
    queue_size: int
    max_reply_chars: int
    feedback_enabled: bool
    require_group_mention: bool
    allowed_chat_ids: set[str]
    verification_token: str
    encrypt_key: str

    @classmethod
    def from_env(cls, *, require_credentials: bool = True) -> "FeishuConfig":
        app_id = os.getenv("FEISHU_APP_ID", "").strip()
        app_secret = os.getenv("FEISHU_APP_SECRET", "").strip()
        backend = os.getenv("FEISHU_BACKEND", "rag").strip().lower()
        if backend not in {"rag", "ollama"}:
            raise ValueError("FEISHU_BACKEND must be either 'rag' or 'ollama'")
        if require_credentials:
            missing = [
                name
                for name, value in (
                    ("FEISHU_APP_ID", app_id),
                    ("FEISHU_APP_SECRET", app_secret),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    "Missing Feishu credentials in .env: " + ", ".join(missing)
                )

        return cls(
            app_id=app_id,
            app_secret=app_secret,
            backend=backend,
            enbot_api_url=os.getenv(
                "ENBOT_API_URL", "http://127.0.0.1:8000/chat"
            ).strip(),
            enbot_api_key=os.getenv("ENBOT_API_KEY", "").strip(),
            ollama_api_url=os.getenv(
                "FEISHU_OLLAMA_API_URL", "http://127.0.0.1:11434/api/chat"
            ).strip(),
            ollama_model=os.getenv(
                "FEISHU_OLLAMA_MODEL",
                os.getenv("OLLAMA_MODEL", "enbot-optuna:7b"),
            ).strip(),
            enbot_timeout_seconds=_env_int("FEISHU_ENBOT_TIMEOUT", 300),
            queue_size=_env_int("FEISHU_QUEUE_SIZE", 16),
            max_reply_chars=_env_int("FEISHU_MAX_REPLY_CHARS", 3500, 200),
            feedback_enabled=_env_bool("FEISHU_FEEDBACK_ENABLED", True),
            require_group_mention=_env_bool(
                "FEISHU_REQUIRE_GROUP_MENTION", True
            ),
            allowed_chat_ids=_csv_set(os.getenv("FEISHU_ALLOWED_CHAT_IDS", "")),
            verification_token=os.getenv("FEISHU_VERIFICATION_TOKEN", "").strip(),
            encrypt_key=os.getenv("FEISHU_ENCRYPT_KEY", "").strip(),
        )


@dataclass(frozen=True)
class MessageJob:
    message_id: str
    chat_id: str
    query: str


@dataclass(frozen=True)
class EnBotResult:
    answer: str
    request_id: str | None = None


@dataclass(frozen=True)
class FeedbackCommand:
    rating: str
    comment: str = ""


class RecentMessageIds:
    """Thread-safe bounded set used to ignore duplicate Feishu deliveries."""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._items: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()

    def add_if_new(self, message_id: str) -> bool:
        with self._lock:
            if message_id in self._items:
                self._items.move_to_end(message_id)
                return False
            self._items[message_id] = None
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
            return True


def extract_text(content: Optional[str], mentions: Optional[Iterable[object]]) -> str:
    """Extract text JSON and remove Feishu mention placeholders."""
    try:
        parsed = json.loads(content or "{}")
    except json.JSONDecodeError:
        return ""

    text = parsed.get("text", "")
    if not isinstance(text, str):
        return ""
    for mention in mentions or ():
        key = getattr(mention, "key", None)
        if key:
            text = text.replace(key, " ")
    return re.sub(r"\s+", " ", text).strip()


def split_reply(text: str, limit: int) -> list[str]:
    """Split a long answer into Feishu-friendly text messages."""
    text = (text or "").strip()
    if not text:
        return ["小英没有生成有效回答，请稍后重试。"]
    chunks: list[str] = []
    while len(text) > limit:
        split_at = max(
            text.rfind("\n", 0, limit + 1),
            text.rfind("。", 0, limit + 1),
            text.rfind("；", 0, limit + 1),
            text.rfind(" ", 0, limit + 1),
        )
        if split_at < limit // 2:
            split_at = limit
        else:
            split_at += 1
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
    if text:
        chunks.append(text)
    return chunks


def extract_enbot_result(payload: object) -> EnBotResult:
    """Validate the EnBot response and retain its feedback correlation ID."""
    if not isinstance(payload, dict):
        raise ValueError("EnBot returned a non-object JSON response")
    if payload.get("status") != "success":
        raise ValueError("EnBot reported an unsuccessful response")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("EnBot response is missing data")
    answer = data.get("response")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("EnBot response is missing answer text")
    request_id = payload.get("request_id")
    if request_id is not None and not isinstance(request_id, str):
        raise ValueError("EnBot response has an invalid request_id")
    return EnBotResult(answer=answer.strip(), request_id=request_id or None)


def extract_enbot_answer(payload: object) -> str:
    """Backward-compatible helper used by the existing self-test."""
    return extract_enbot_result(payload).answer


def parse_feedback_command(text: str) -> FeedbackCommand | None:
    """Recognize explicit text feedback without intercepting normal questions."""
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if normalized in {"有帮助", "有用", "好评", "👍", "👍🏻", "👍🏼"}:
        return FeedbackCommand(rating="up")
    match = re.fullmatch(
        r"(?:没帮助|无帮助|差评|👎|👎🏻|👎🏼)(?:[\s：:，,。-]+(.*))?",
        normalized,
    )
    if match:
        return FeedbackCommand(rating="down", comment=(match.group(1) or "").strip())
    return None


def extract_ollama_answer(payload: object) -> str:
    """Validate a non-streaming Ollama /api/chat response."""
    if not isinstance(payload, dict):
        raise ValueError("Ollama returned a non-object JSON response")
    message = payload.get("message")
    if not isinstance(message, dict):
        raise ValueError("Ollama response is missing message")
    answer = message.get("content")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Ollama response is missing answer text")
    return answer.strip()


def feedback_prompt() -> str:
    return (
        "\n\n\u2014\u2014\n\u8fd9\u6761\u56de\u7b54\u662f\u5426\u6709"
        "\u5e2e\u52a9\uff1f\u8bf7\u56de\u590d\u201c\u6709\u5e2e\u52a9"
        "\u201d\uff0c\u6216\u201c\u6ca1\u5e2e\u52a9 + \u539f\u56e0"
        "\u201d\u3002"
    )


def feedback_no_previous_message() -> str:
    return (
        "\u6682\u65f6\u627e\u4e0d\u5230\u4e0a\u4e00\u6761\u56de\u7b54"
        "\uff0c\u8bf7\u5148\u5411\u5c0f\u82f1\u63d0\u95ee\u540e\u518d"
        "\u63d0\u4ea4\u53cd\u9988\u3002"
    )


def feedback_ack_message(rating: str) -> str:
    if rating == "up":
        return (
            "\u611f\u8c22\u53cd\u9988\uff0c\u5df2\u8bb0\u5f55\u4e3a"
            "\u201c\u6709\u5e2e\u52a9\u201d\u3002"
        )
    return (
        "\u611f\u8c22\u53cd\u9988\uff0c\u5df2\u8bb0\u5f55\u4e3a\u201c"
        "\u6ca1\u5e2e\u52a9\u201d\uff0c\u540e\u7eed\u4f1a\u8fdb\u5165"
        "\u4eba\u5de5\u590d\u6838\u3002"
    )


class FeishuEnBotBridge:
    def __init__(self, config: FeishuConfig, api_client: lark.Client):
        self.config = config
        self.api_client = api_client
        self.jobs: queue.Queue[MessageJob] = queue.Queue(maxsize=config.queue_size)
        self.seen = RecentMessageIds()
        self.http = requests.Session()
        self._last_request_ids: OrderedDict[str, str] = OrderedDict()
        self._feedback_lock = threading.Lock()
        self.worker = threading.Thread(
            target=self._worker_loop,
            name="feishu-enbot-worker",
            daemon=True,
        )

    def start(self) -> None:
        self.worker.start()

    def handle_message(self, event: P2ImMessageReceiveV1) -> None:
        """Feishu event callback; enqueue quickly and return to acknowledge it."""
        if event.event is None or event.event.message is None:
            LOGGER.warning("Ignored an event without message data")
            return

        message = event.event.message
        message_id = message.message_id or ""
        chat_id = message.chat_id or ""
        if not message_id or not chat_id:
            LOGGER.warning("Ignored a message without message_id or chat_id")
            return
        if not self.seen.add_if_new(message_id):
            LOGGER.info("Ignored duplicate message: %s", message_id)
            return
        if self.config.allowed_chat_ids and chat_id not in self.config.allowed_chat_ids:
            LOGGER.warning("Ignored a message from a chat outside the allowlist")
            return
        if message.message_type != "text":
            self._safe_reply(message_id, "演示版目前仅支持文字消息。")
            return
        if (
            message.chat_type == "group"
            and self.config.require_group_mention
            and not message.mentions
        ):
            return

        query = extract_text(message.content, message.mentions)
        if not query:
            self._safe_reply(message_id, "请在 @小英 后输入你的问题。")
            return

        try:
            self.jobs.put_nowait(
                MessageJob(message_id=message_id, chat_id=chat_id, query=query)
            )
            LOGGER.info("Queued message %s (queue=%d)", message_id, self.jobs.qsize())
        except queue.Full:
            LOGGER.warning("Work queue is full")
            self._safe_reply(message_id, "当前问题较多，请稍后重新发送。")

    def _worker_loop(self) -> None:
        while True:
            job = self.jobs.get()
            try:
                feedback = parse_feedback_command(job.query)
                if (
                    feedback is not None
                    and self.config.feedback_enabled
                    and self.config.backend == "rag"
                ):
                    answer = self._submit_feedback(job.chat_id, feedback)
                else:
                    result = self._ask_model(job.query, job.chat_id)
                    if result.request_id:
                        self._remember_request_id(job.chat_id, result.request_id)
                    answer = result.answer
                    if self.config.feedback_enabled and result.request_id:
                        answer += feedback_prompt()
                self._reply(job.message_id, answer)
            except requests.Timeout:
                LOGGER.exception("EnBot request timed out")
                self._safe_reply(
                    job.message_id,
                    "小英处理问题超时，请确认本地模型正在运行后重试。",
                )
            except Exception:
                LOGGER.exception("Failed to process Feishu message")
                self._safe_reply(
                    job.message_id,
                    "小英暂时无法连接本地知识库服务，请稍后重试。",
                )
            finally:
                self.jobs.task_done()

    def _remember_request_id(self, chat_id: str, request_id: str) -> None:
        with self._feedback_lock:
            self._last_request_ids[chat_id] = request_id
            self._last_request_ids.move_to_end(chat_id)
            while len(self._last_request_ids) > 1000:
                self._last_request_ids.popitem(last=False)

    def _last_request_id(self, chat_id: str) -> str | None:
        with self._feedback_lock:
            return self._last_request_ids.get(chat_id)

    def _feedback_url(self) -> str:
        chat_url = self.config.enbot_api_url.rstrip("/")
        if chat_url.endswith("/v1/chat"):
            return chat_url[:-5] + "/feedback"
        if chat_url.endswith("/chat"):
            return chat_url[:-5] + "/v1/feedback"
        return chat_url + "/v1/feedback"

    def _submit_feedback(
        self, chat_id: str, command: FeedbackCommand
    ) -> str:
        request_id = self._last_request_id(chat_id)
        if not request_id:
            return feedback_no_previous_message()
        headers = (
            {"X-API-Key": self.config.enbot_api_key}
            if self.config.enbot_api_key
            else None
        )
        response = self.http.post(
            self._feedback_url(),
            json={
                "request_id": request_id,
                "rating": command.rating,
                "categories": [],
                "comment": command.comment,
                "expected_answer": "",
            },
            headers=headers,
            timeout=(5, self.config.enbot_timeout_seconds),
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("status") != "success":
            raise ValueError("EnBot feedback endpoint returned an invalid response")
        return feedback_ack_message(command.rating)

    def _ask_model(self, query: str, session_id: str) -> EnBotResult:
        if self.config.backend == "ollama":
            return self._ask_ollama(query)
        return self._ask_enbot(query, session_id)

    def _ask_enbot(self, query: str, session_id: str) -> EnBotResult:
        headers = (
            {"X-API-Key": self.config.enbot_api_key}
            if self.config.enbot_api_key
            else None
        )
        response = self.http.post(
            self.config.enbot_api_url,
            json={
                "query": query,
                "session_id": session_id,
                "evaluate": False,
                "debug": False,
            },
            headers=headers,
            timeout=(5, self.config.enbot_timeout_seconds),
        )
        response.raise_for_status()
        return extract_enbot_result(response.json())

    def _ask_ollama(self, query: str) -> EnBotResult:
        response = self.http.post(
            self.config.ollama_api_url,
            json={
                "model": self.config.ollama_model,
                "messages": [{"role": "user", "content": query}],
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=(5, self.config.enbot_timeout_seconds),
        )
        response.raise_for_status()
        payload = response.json()
        eval_count = payload.get("eval_count", 0)
        eval_duration = payload.get("eval_duration", 0)
        prompt_eval_count = payload.get("prompt_eval_count", 0)
        prompt_eval_duration = payload.get("prompt_eval_duration", 0)
        total_duration = payload.get("total_duration", 0)

        generation_rate = (
            eval_count / (eval_duration / 1_000_000_000)
            if eval_count and eval_duration
            else 0.0
        )
        prompt_rate = (
            prompt_eval_count / (prompt_eval_duration / 1_000_000_000)
            if prompt_eval_count and prompt_eval_duration
            else 0.0
        )
        LOGGER.info(
            "Ollama metrics: output_tokens=%s, generation=%.2f tok/s, "
            "prompt=%.2f tok/s, total=%.2f s",
            eval_count,
            generation_rate,
            prompt_rate,
            total_duration / 1_000_000_000 if total_duration else 0.0,
        )
        return EnBotResult(answer=extract_ollama_answer(payload))

    def _reply(self, message_id: str, answer: str) -> None:
        for index, chunk in enumerate(
            split_reply(answer, self.config.max_reply_chars), start=1
        ):
            digest = hashlib.sha256(
                f"{message_id}:{index}".encode("utf-8")
            ).hexdigest()[:24]
            body = (
                ReplyMessageRequestBody.builder()
                .content(json.dumps({"text": chunk}, ensure_ascii=False))
                .msg_type("text")
                .uuid(f"enbot-{digest}")
                .build()
            )
            request = (
                ReplyMessageRequest.builder()
                .message_id(message_id)
                .request_body(body)
                .build()
            )
            response = self.api_client.im.v1.message.reply(request)
            if not response.success():
                raise RuntimeError(
                    "Feishu reply failed: "
                    f"code={response.code}, msg={response.msg}, "
                    f"log_id={response.get_log_id()}"
                )

    def _safe_reply(self, message_id: str, text: str) -> None:
        try:
            self._reply(message_id, text)
        except Exception:
            LOGGER.exception("Failed to send fallback reply")


def run_self_test() -> None:
    class Mention:
        key = "@_user_1"

    assert extract_text('{"text":"@_user_1  测试问题"}', [Mention()]) == "测试问题"
    assert split_reply("abcdef", 3) == ["abc", "def"]
    assert (
        extract_enbot_answer(
            {"status": "success", "data": {"response": "测试答案"}}
        )
        == "测试答案"
    )
    assert (
        extract_ollama_answer(
            {"message": {"role": "assistant", "content": "direct answer"}}
        )
        == "direct answer"
    )
    print("Feishu bridge self-test: PASS")


def check_configuration(config: FeishuConfig) -> int:
    print("Feishu SDK import: PASS")
    print(f"FEISHU_APP_ID configured: {bool(config.app_id)}")
    print(f"FEISHU_APP_SECRET configured: {bool(config.app_secret)}")
    print(f"Backend: {config.backend}")
    try:
        if config.backend == "ollama":
            print(f"Ollama model: {config.ollama_model}")
            tags_url = config.ollama_api_url.rsplit("/api/", 1)[0] + "/api/tags"
            response = requests.get(tags_url, timeout=5)
            response.raise_for_status()
            model_names = {
                item.get("name")
                for item in response.json().get("models", [])
                if isinstance(item, dict)
            }
            if config.ollama_model not in model_names:
                print("Ollama model available: FAIL")
                return 2
            print("Ollama model available: PASS")
        else:
            print(f"EnBot API: {config.enbot_api_url}")
            base_url = config.enbot_api_url.rsplit("/", 1)[0] + "/"
            response = requests.get(base_url, timeout=5)
            response.raise_for_status()
            print("EnBot API reachable: PASS")
    except requests.RequestException as exc:
        print(f"Selected backend reachable: FAIL ({exc})")
        return 2
    return 0 if config.app_id and config.app_secret else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="EnBot Feishu long-connection bridge")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=os.getenv("FEISHU_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.self_test:
        run_self_test()
        return 0

    try:
        config = FeishuConfig.from_env(require_credentials=not args.check)
    except (TypeError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.check:
        return check_configuration(config)

    api_client = (
        lark.Client.builder()
        .app_id(config.app_id)
        .app_secret(config.app_secret)
        .log_level(lark.LogLevel.INFO)
        .build()
    )
    bridge = FeishuEnBotBridge(config, api_client)
    bridge.start()

    event_handler = (
        lark.EventDispatcherHandler.builder(
            config.encrypt_key,
            config.verification_token,
            lark.LogLevel.INFO,
        )
        .register_p2_im_message_receive_v1(bridge.handle_message)
        .build()
    )
    ws_client = lark.ws.Client(
        config.app_id,
        config.app_secret,
        log_level=lark.LogLevel.INFO,
        event_handler=event_handler,
        auto_reconnect=True,
    )

    print("=" * 60)
    print("EnBot Feishu long connection is starting")
    print(f"Backend: {config.backend}")
    if config.backend == "ollama":
        print(f"Ollama model: {config.ollama_model}")
        print(f"Ollama API: {config.ollama_api_url}")
    else:
        print(f"EnBot API: {config.enbot_api_url}")
    print("Credentials loaded: yes (values are not printed)")
    print("After the connection succeeds, click Verify in Feishu Open Platform.")
    print("Press Ctrl+C to stop.")
    print("=" * 60)
    try:
        ws_client.start()
    except KeyboardInterrupt:
        print("Feishu bridge stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
