"""Local Gradio interface for the EnBot text and multimodal APIs."""

from __future__ import annotations

import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

TEXT_API_URL = os.getenv("GRADIO_TEXT_API_URL", "http://127.0.0.1:8000/v1/chat")
VISION_API_URL = os.getenv(
    "GRADIO_VISION_API_URL",
    "http://127.0.0.1:8011/v1/diagnose",
)
REQUEST_TIMEOUT = int(os.getenv("GRADIO_REQUEST_TIMEOUT", "300"))
HTTP = requests.Session()


def _headers(api_key: str) -> dict[str, str] | None:
    return {"X-API-Key": api_key} if api_key else None


def _source_value(source: object) -> str:
    if isinstance(source, str):
        return source.strip()
    if isinstance(source, dict):
        return str(
            source.get("url")
            or source.get("source")
            or source.get("title")
            or ""
        ).strip()
    return ""


def format_sources(sources: object) -> str:
    if not isinstance(sources, list):
        return "暂无来源"
    values: list[str] = []
    seen: set[str] = set()
    for source in sources:
        value = _source_value(source)
        if value and value not in seen:
            seen.add(value)
            values.append(value)
    if not values:
        return "暂无来源"
    return "\n".join(f"- {value}" for value in values)


def _error_message(exc: Exception) -> tuple[str, str, dict[str, Any]]:
    return (
        f"请求失败：{exc}",
        "请确认相关本地服务已经启动。",
        {"status": "error", "error": str(exc)},
    )


def ask_text(query: str, session_id: str) -> tuple[str, str, dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return "请输入问题。", "暂无来源", {"status": "waiting"}
    started = time.perf_counter()
    try:
        response = HTTP.post(
            TEXT_API_URL,
            json={
                "query": query,
                "session_id": (session_id or "gradio-text").strip(),
                "evaluate": False,
                "debug": False,
            },
            headers=_headers(os.getenv("ENBOT_API_KEY", "").strip()),
            timeout=(5, REQUEST_TIMEOUT),
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "success" or not isinstance(payload.get("data"), dict):
            raise ValueError("文本服务返回了无效响应")
        data = payload["data"]
        answer = str(data.get("response") or "").strip()
        if not answer:
            raise ValueError("文本服务没有返回回答")
        metadata = {
            "status": "success",
            "request_id": payload.get("request_id"),
            "retrieval_backend": data.get("retrieval_backend"),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        return answer, format_sources(data.get("sources")), metadata
    except (requests.RequestException, ValueError, TypeError) as exc:
        return _error_message(exc)


def diagnose_image(
    image_path: str | None,
    question: str,
    session_id: str,
) -> tuple[str, str, dict[str, Any]]:
    if not image_path:
        return "请先上传图片。", "暂无来源", {"status": "waiting"}
    path = Path(image_path)
    if not path.is_file():
        return "图片文件不存在。", "暂无来源", {"status": "error"}

    started = time.perf_counter()
    try:
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        with path.open("rb") as image_file:
            response = HTTP.post(
                VISION_API_URL,
                files={"image": (path.name, image_file, media_type)},
                data={
                    "question": (question or "请分析图片中的设备状态或故障。").strip(),
                    "session_id": (session_id or "gradio-vision").strip(),
                    "debug": "false",
                },
                headers=_headers(os.getenv("VISION_API_KEY", "").strip()),
                timeout=(10, REQUEST_TIMEOUT),
            )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ValueError("视觉服务返回了无效响应")
        answer = str(data.get("answer") or "").strip()
        if not answer:
            raise ValueError("视觉服务没有返回诊断结果")
        metadata = {
            "status": payload.get("status"),
            "request_id": payload.get("request_id"),
            "retrieval_backend": data.get("retrieval_backend"),
            "total_seconds": data.get("total_seconds"),
            "client_elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        return answer, format_sources(data.get("sources")), metadata
    except (OSError, requests.RequestException, ValueError, TypeError) as exc:
        return _error_message(exc)


def check_services() -> dict[str, Any]:
    result: dict[str, Any] = {}
    checks = {
        "text_api": "http://127.0.0.1:8000/health/ready",
        "vision_api": "http://127.0.0.1:8011/health/ready",
    }
    for name, url in checks.items():
        try:
            response = HTTP.get(url, timeout=10)
            response.raise_for_status()
            result[name] = {"ready": True, "details": response.json()}
        except (requests.RequestException, ValueError) as exc:
            result[name] = {"ready": False, "error": str(exc)}
    return result


def create_demo():
    import gradio as gr

    with gr.Blocks(title="小英 · 家庭储能智能助手") as demo:
        gr.Markdown(
            "# 小英 · 家庭储能智能助手\n"
            "本地运行的微调模型、Milvus RAG 与多模态故障诊断演示。"
        )
        with gr.Row():
            status_button = gr.Button("检查本地服务", variant="secondary")
            status_output = gr.JSON(label="服务状态")
        status_button.click(check_services, outputs=status_output)

        with gr.Tabs():
            with gr.Tab("文字问答"):
                text_query = gr.Textbox(
                    label="问题",
                    placeholder="例如：怎么设置负电价限制馈网？",
                    lines=3,
                )
                text_session = gr.Textbox(
                    label="会话编号",
                    value="gradio-text-demo",
                )
                text_button = gr.Button("发送问题", variant="primary")
                text_answer = gr.Markdown(label="回答")
                text_sources = gr.Markdown(label="参考来源")
                text_metadata = gr.JSON(label="运行信息")
                text_button.click(
                    ask_text,
                    inputs=[text_query, text_session],
                    outputs=[text_answer, text_sources, text_metadata],
                )

            with gr.Tab("图片诊断"):
                image_input = gr.Image(label="上传设备或故障图片", type="filepath")
                image_question = gr.Textbox(
                    label="希望系统分析什么",
                    value="请识别图片中的设备、故障或告警信息，并给出有依据的排查建议。",
                    lines=2,
                )
                image_session = gr.Textbox(
                    label="会话编号",
                    value="gradio-vision-demo",
                )
                image_button = gr.Button("开始诊断", variant="primary")
                image_answer = gr.Markdown(label="诊断结果")
                image_sources = gr.Markdown(label="参考来源")
                image_metadata = gr.JSON(label="运行信息")
                image_button.click(
                    diagnose_image,
                    inputs=[image_input, image_question, image_session],
                    outputs=[image_answer, image_sources, image_metadata],
                )

        gr.Markdown(
            "演示页面默认仅在本机开放，不使用 Gradio 公网分享功能。"
        )
    return demo


def main() -> None:
    demo = create_demo()
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
