"""Small non-RAG tools available to the fallback LangChain agent."""

from datetime import datetime

import numexpr as ne
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool


@tool
def search(query: str) -> str:
    """Search the public web when the local product knowledge base is not applicable."""
    try:
        result = DuckDuckGoSearchRun().run(query)
        return result or "未搜索到相关结果。"
    except Exception as exc:
        return f"搜索工具暂时不可用：{exc}"


@tool
def get_weather(city: str) -> str:
    """Return current weather for a city."""
    try:
        response = requests.get(f"https://wttr.in/{city}?format=%C+%t+%w", timeout=10)
        response.raise_for_status()
        return response.text.strip()
    except Exception as exc:
        return f"无法获取{city}的天气：{exc}"


@tool
def get_current_time() -> str:
    """Return the current local date and time."""
    return datetime.now().strftime("当前时间：%Y年%m月%d日 %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """Evaluate a numeric expression."""
    try:
        return f"计算结果：{ne.evaluate(expression)}"
    except Exception as exc:
        return f"表达式有误：{exc}"
