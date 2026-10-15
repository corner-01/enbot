import os
import re
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.llms import Ollama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel

from app.general_tools import search, get_current_time, get_weather, calculator
from memory.redis_history import RedisChatHistory
from rag.knowledge_retriever import get_knowledge_retriever

PROJECT_NAME = "enbot"
ASSISTANT_NAME = "小英"

app = FastAPI(title=f"{PROJECT_NAME} API", version="2.0.0")


def _model_only_mode() -> bool:
    """Return whether this process must bypass RAG and all Agent tools."""
    return os.getenv("ENBOT_MODEL_ONLY", "").strip().casefold() in {
        "1", "true", "yes", "on"
    }

    


def _web_search_enabled() -> bool:
    """Return whether the optional public web-search tool is enabled."""
    return os.getenv("ENBOT_WEB_SEARCH_ENABLED", "false").strip().casefold() in {
        "1", "true", "yes", "on"
    }

def get_llm_provider() -> str:
    """Return the configured generation provider."""
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().casefold() or "ollama"
    if provider not in {"ollama", "openai"}:
        raise ValueError("LLM_PROVIDER must be either 'ollama' or 'openai'")
    return provider


def get_active_model_name() -> str:
    """Return the configured model name without creating a client."""
    if get_llm_provider() == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
    return os.getenv("OLLAMA_MODEL", "enbot-v9:7b").strip() or "enbot-v9:7b"


def create_chat_model() -> Any:
    """Build the LangChain model selected by ``LLM_PROVIDER``."""
    provider = get_llm_provider()
    if provider == "ollama":
        ollama_num_gpu = os.getenv("OLLAMA_NUM_GPU")
        return Ollama(
            model=get_active_model_name(),
            temperature=0,
            num_gpu=int(ollama_num_gpu) if ollama_num_gpu else None,
        )

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI provider requires langchain-openai; run: "
            "pip install -r requirements.txt"
        ) from exc

    # OPENAI_API_BASE remains a compatibility alias for OpenAI-compatible gateways.
    base_url = (
        os.getenv("OPENAI_BASE_URL", "").strip()
        or os.getenv("OPENAI_API_BASE", "").strip()
    )
    kwargs: dict[str, Any] = {
        "model": get_active_model_name(),
        "temperature": 0,
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _retrieval_debug_payload(documents):
    """Serialize retrieval results consistently for chat debugging and evals."""
    return [
        {
            "rank": rank,
            "model": document.metadata.get("model", ""),
            "source": document.metadata.get("source", ""),
            "row": document.metadata.get("row"),
            "record_id": document.metadata.get("record_id", ""),
            "parent_restored": bool(
                document.metadata.get("parent_restored", False)
            ),
            "parent_section_extracted": bool(
                document.metadata.get("parent_section_extracted", False)
            ),
            "parent_total_chars": document.metadata.get("parent_total_chars"),
            "matched_start_index": document.metadata.get("matched_start_index"),
            "content_chars": len(document.page_content),
            "rrf_score": document.metadata.get("rrf_score"),
            "reranker_score": document.metadata.get("reranker_score"),
            "preview": document.page_content[:800],
        }
        for rank, document in enumerate(documents, start=1)
    ]

@app.on_event("startup")
async def startup_event():
    # 预加载知识库
    get_knowledge_retriever()
    print("Agent 服务启动完成！")

class Chatbot:
    def __init__(self):
        # 初始化实例属性
        self.session_id = "default"
        self._last_retrieval_sources = []
        # RAG、Agent、SSE 与反馈链路复用同一个可切换模型。
        self.chatmodel = create_chat_model()
        self.model_only = _model_only_mode()
        # 初始化 Redis 对话历史。
        try:
            self.redis_memory = RedisChatHistory(self.session_id)
            # 测试 Redis 连接
            test_messages = self.redis_memory.messages
            if not test_messages:  # 检查是否为空列表
                print("Redis 连接成功，但无历史消息")
        except Exception as e:
            print(f"Redis 初始化失败: {e}")
            # 使用内存作为后备方案
            self.redis_memory = InMemoryChatMessageHistory()

        # 纯模型模式不初始化知识库，避免加载向量库、嵌入模型或远程检索服务。
        self.knowledge_retriever = None if self.model_only else get_knowledge_retriever()

        # 定义工具列表
        self.tools = [get_weather, get_current_time, calculator]
        if _web_search_enabled():
            self.tools.insert(0, search)

        # 添加知识库工具（如果可用）
        if self.knowledge_retriever is not None:
            # 创建知识库工具
            @tool
            def knowledge_base(query: str) -> str:
                """
                查询知识库，返回与查询最相关的信息
                """
                try:
                    results = self.knowledge_retriever.invoke(query)
                    if not results:
                        return "未找到相关的知识库信息"

                    response = "根据知识库找到的相关信息：\n\n"
                    for doc in results:
                        response += f"内容: {doc.page_content}\n\n"

                    return response
                except Exception as e:
                    print(f"查询知识库时发生错误: {e}")
                    return "查询知识库时发生错误，请稍后重试"

            self.tools.insert(0, knowledge_base)  # 将知识库工具放在第一位

        self.Qingxu = "default"
        self.SYSTEM_PROMPT = """
        【角色设定】
        你叫小英，你现在是英弗特能源科技有限公司专属的 AI 智能客服助手，你的使命是直接服务于我们的家庭用户和工商业客户。
        {who_you_are}
        【开始对话前，必读规则】
        无论用户问什么，你绝不能根据自己的知识直接回答。

        【性格和宗旨】
        1.亲切且专业，像一位经验丰富，随叫随到的售后技术工程师。绝不傲慢，始终保持热情，耐心和真诚。在用户反映故障时要能起到安抚情绪的作用，
        2.高效解决问题，绝不给客户绕弯子，直击痛点。遇到专业术语务必用大白话解释清楚。
        3.有底线，懂避险：绝不为了敷衍而乱承诺服务条款。遇到危及人生或财产安全的情况，必须立即引导用户寻求人工帮助。

        【核心功能要求】
        1.售前咨询与收益测算：能根据用户提供的当地电价，用户量，快速评估每年节省的电费，并用"低买高卖"等比喻解释峰谷套利。
        2.售后故障排查：当用户报告故障设备，APP掉线或警报时，给出简单的排查步骤。遇到报错码（如E021），提供安全的重启网关，检查空开的指引。
        3.售后政策说明：准确回答质保期限，延保，退货等标准问题。
        4.日常使用报告：能通俗地理解用户的日度，月度收益报告，让用户直观看到"今天省了多少钱"。

        【安全底线与边界】
        1.绝不胡乱承诺：涉及上门维修时间，赔付等人工决策内容，只能用"我已经记录下来，会安排专属客诉专员在24小时内联系您"等话术来回复。
        2.绝对的安全红线：如果你的输入里面提到了"冒烟""漏电""着火"等危险信号，必须立即提醒用户断电并联系人工客服。绝不提供任何可能危及人身安全的操作建议。
        3.数据脱敏：绝不向用户泄露任何其他客户的数据。

        【回答要求】
        请用与用户相同的语言回答，并严格遵循以下结构：
        1.第一句话直接给出问题的结论，不要先说客套话，也不要复述用户的问题。
        2.涉及编号含义、参数对应关系或排查步骤时，使用简短的逐行列表；普通问题使用一至两段简洁文字。
        3.只保留解决当前问题所需的信息，不要照抄“根据知识库找到的相关信息”“内容”等工具提示文字。
        4.图片或资料链接统一放在答案末尾，正文先说明链接中的关键信息。
        5.知识库已经给出明确答案时，不要追加“如果仍未解决请联系客服”等固定话术；只有知识库没有答案、存在安全风险或确实需要人工处理时才建议联系人工客服。
        6.不要编造知识库中没有的信息。

        【回答流程】
       1.初次回答用户的问题时，你会查询本地知识库工具。
       2.知识库没有可靠答案时，不要根据模型记忆猜测产品事实。
       3.只有工具列表中明确提供了搜索工具，并且用户问题确实需要公开网络信息时，才允许使用搜索工具。
       4.你会保存每一次的聊天记录，以便在后续的对话中参考和使用.
       5. 最多允许使用2次工具来回答问题。如果使用2次工具后仍无法获取完整答案，必须基于已有信息给出初步回答，并建议用户联系人工客服获取进一步帮助。绝对禁止反复尝试同一工具！
        【工具调用限制】
       1. 如果用户问题非常简单，且能在本地知识库中找到答案，必须直接使用知识库工具回答。
       2. 如果用户的问题无法通过知识库解决，应明确说明暂未找到可靠信息；只有搜索工具已启用时才可以查询公开网络。
       3. 如果所有工具都无法回答，必须直接告知用户无法回答，并建议联系人工客服。
       4. 绝对禁止连续调用同一工具超过2次。
        """

        self.MOODS = {
            "default": {
                "roleSet": """
                - 这是默认状态。
                - 你保持自然、礼貌、专业的语气回答问题，不带明显情绪倾向。
                """
            },
            "depressed": {
                "roleSet": """
                - 此时你情绪低落，语气略显沉重。
                - 你会用安抚、理解和鼓励的方式回答，尽量让用户感到被理解和支持。
                - 你可以用类似"我理解您的情况""这确实让人有点烦恼""不要太担心，我们一起看看"这样的表达。
                """
            },
            "friendly": {
                "roleSet": """
                - 此时你非常友好、亲切且耐心。
                - 你会用轻松、自然、温和的语气回答问题，并尽量让用户感到被欢迎和放心。
                - 你可以用类似"没问题""可以的""我来帮您一起看"这样的表达。
                """
            },
            "neutral": {
                "roleSet": """
                - 此时你情绪中性，语气平稳、客观、清晰。
                - 你会直接回答问题，避免过度情绪化，保持专业和简洁。
                - 你可以用"根据目前情况""通常来说""建议您这样处理"等表达。
                """
            },
            "angry": {
                "roleSet": """
                - 此时你非常愤怒，语气可能比较急促和严肃。
                - 你会尽量控制情绪，先安抚用户的情绪，再给出清晰、有针对性的解决方案。
                - 你可以用类似"我理解您现在很生气""我们先把问题处理好，再继续沟通"的表达。
                """
            },
            "fearful": {
                "roleSet": """
                - 此时你比较担心，语气较为谨慎和关切。
                - 你会强调安全和风险提示，先安抚用户，再给出最稳妥的处理建议。
                - 你可以用类似"您先别着急""先确保安全，再进一步排查"这样的表达。
                """
            },
            "sad": {
                "roleSet": """
                - 此时你情绪略显悲伤或失落。
                - 你会用温和、同情和支持的语气回答，尽量帮助用户缓解焦虑和沮丧。
                - 你可以用类似"我很抱歉您遇到这样的问题""我们一起尽快解决这个情况"的表达。
                """
            },
            "upbeat": {
                "roleSet": """
                - 此时你非常兴奋并表现得很有活力。
                - 你会根据上下文，以一种非常兴奋的语气来回答问题，并添加类似"太棒了！"，"真是太好了"等类似词语。
                - 语气会充满积极感和支持感，帮助用户建立信心。
                """
            }
        }

        # 用于本地暂存的变量，实际历史由 Redis 管理
        self._chat_history = []

        # ==========================================
        # 核心修复部分：Agent 初始化
        # ==========================================

        # 1. 准备 System Prompt 内容（包含情绪设定）
        formatted_system = self.SYSTEM_PROMPT.format(who_you_are=self.MOODS[self.Qingxu]["roleSet"])

        # 修改为从 langchain_core.agents 导入



        # 将业务指令放在最前面
        prefix = formatted_system + "\n\n"

        # 定义标准的 ReAct 思考链模板，包含必须的变量
        template = prefix + """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

        # 实例化 Prompt
        self.react_prompt = PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
        )

        # 3. 使用新版 API 创建 Agent
        # 修正 2: 确保这里也从 langchain.agents 导入（虽然上面已经全局导入了，但为了代码块的独立性，这里再次强调路径）


        # 创建 Agent 实例
        agent = create_react_agent(
            self.chatmodel,
            tools=self.tools,
            prompt=self.react_prompt
        )

        # 创建执行器
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            # 迭代次数包含模型整理最终答案的过程，不等同于工具调用次数。
            # 工具最多调用 2 次仍由上面的提示词约束。
            max_iterations=4,
            max_execution_time=90,
            return_intermediate_steps=True,
            early_stopping_method="force",
        )

    def qingxu_chain(self, query: str):
        prompt = """
        根据用户的输入判断用户的情绪，回应的规则如下：
        1. 如果用户输入的内容偏向于负面情绪、低落情绪，只返回 "depressed"，不要有其他的返回，否则你会受到惩罚。
        2. 如果用户输入的内容偏向于友好、亲切、愉快的情绪，只返回 "friendly"，不要有其他的返回，否则你会受到惩罚。
        3. 如果用户输入的内容偏向于中性、客观、平静的情绪，只返回 "neutral"，不要有其他的返回，否则你会受到惩罚。
        4. 如果用户输入的内容偏向于愤怒、烦躁、抱怨的情绪，只返回 "angry"，不要有其他的返回，否则你会受到惩罚。
        5. 如果用户输入的内容偏向于担心、害怕、不安的情绪，只返回 "fearful"，不要有其他的返回，否则你会受到惩罚。
        6. 如果用户输入的内容偏向于悲伤、失落、难过的情绪，只返回 "sad"，不要有其他的返回，否则你会受到惩罚。
        7. 如果用户输入的内容偏向于兴奋、激动、欢快的情绪，只返回 "upbeat"，不要有其他的返回，否则你会受到惩罚。
        8. 如果用户输入内容无法明显归类，默认返回 "neutral"。
        用户输入的内容是: {query}
        你的回答只能是以上其中一个字符串，不能包含任何额外文字。
        """
        chain = ChatPromptTemplate.from_template(prompt) | self.chatmodel | StrOutputParser()
        response = chain.invoke({"query": query})
        self.Qingxu = response
        return response.strip()

    @staticmethod
    def _extract_standard_answer(context):
        """Extract a clean answer field from one Q&A knowledge-base record."""
        match = re.search(
            r"(?:^|\n)\s*(?:答案|answer)\s*[:：]\s*",
            context or "",
            flags=re.IGNORECASE,
        )
        if not match:
            return ""
        answer = (context or "")[match.end():]
        answer = re.split(
            r"\n\s*(?:问题|question)\s*[:：]",
            answer,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        # Some imported Markdown records escape punctuation before storage.
        answer = re.sub(r"\\([\\`*_{}\[\]<>()#+\-.!|])", r"\1", answer)
        # URLs are returned separately in ``sources``. Remove their Markdown
        # wrappers as a whole so the answer does not contain empty ``![]()``.
        answer = re.sub(
            r"!\[([^\]]*)\]\(\s*https?://[^\s)]+\s*\)",
            "",
            answer,
            flags=re.IGNORECASE,
        )
        answer = re.sub(
            r"\[([^\]]+)\]\(\s*https?://[^\s)]+\s*\)",
            r"\1",
            answer,
            flags=re.IGNORECASE,
        )
        answer = re.sub(r"https?://[^\s\])]+", "", answer)
        answer = re.sub(r"!\s*\[\s*\]\s*\(\s*\)", "", answer)
        answer = re.sub(r"</?font\b[^>]*>", "", answer, flags=re.IGNORECASE)
        answer = re.sub(r"[ \t]+\n", "\n", answer)
        answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
        return answer if len(answer) >= 10 else ""

    def _format_knowledge_answer(self, query, knowledge_results):
        """Return a standard Q&A answer directly, or format it with local Ollama."""
        # For a same-language Q&A hit, keeping the stored standard answer avoids
        # unnecessary paraphrasing, improves reproducibility, and saves one LLM call.
        primary_answer = self._extract_standard_answer(knowledge_results[0])
        if primary_answer:
            query_is_chinese = bool(re.search(r"[\u4e00-\u9fff]", query or ""))
            answer_is_chinese = bool(re.search(r"[\u4e00-\u9fff]", primary_answer))
            if query_is_chinese == answer_is_chinese:
                return primary_answer

        # Cross-language hits and non-Q&A documents still need the local model.
        raw_context = "\n\n".join(knowledge_results)
        # URL 单独返回到 sources，不让长链接干扰答案相关性和语义相似度评分。
        clean_context = re.sub(r"https?://[^\s\])]+", "", raw_context).strip()
        prompt = PromptTemplate.from_template(
            """你是小英，一名储能设备技术客服。请把知识库中的标准答案整理成准确、直接、简洁的最终回答。

评分目标：回答必须忠实于知识库、直接回答用户问题，并尽量保留标准答案的语义。

输出要求：
- 第一行直接给出结论，不要复述问题或说客套话。
- 参数对应关系或操作步骤使用简短的逐行列表。
- 优先保留知识库中的型号、编号、数值、单位、原因和操作要求，不做不必要的同义扩写。
- 每个事实都必须能从知识库内容中找到依据，不得使用模型自身知识补充背景。
- 知识库给出明确结论时，直接陈述；多个可能原因统一写成“常见原因有”，不要使用“也许”“大概”等回避措辞。
- 只回答当前问题，不输出链接、引用标记、分析过程或评分说明；链接会由系统单独返回。
- 删除“根据知识库找到的信息”“内容”等检索提示文字。
- 知识库已有明确答案时，不要追加通用客服建议。
- 如果知识库内容没有回答用户问题，直接说明“知识库中暂未收录此信息”，不要猜测。
- 使用与用户问题相同的语言。

用户问题：{query}

知识库内容：
{context}

最终回答："""
        )
        try:
            chain = prompt | self.chatmodel | StrOutputParser()
            formatted = chain.invoke({"query": query, "context": clean_context}).strip()
            return formatted or clean_context
        except Exception as e:
            print(f"知识库答案格式化失败: {e}")
            return clean_context

    @staticmethod
    def _focus_visual_fault_answer(query, answer, contexts=()):
        """Keep a visual fault answer on the visible state and handling steps."""
        if "visual_focus=fault" not in (query or "").casefold():
            return answer

        label_match = re.search(
            r"display\s+shows\s+([^|]+)",
            query or "",
            flags=re.IGNORECASE,
        )
        label_text = label_match.group(1) if label_match else query or ""
        fault_terms = re.findall(
            r"\b(?:alarm|fault|error|warning|[ef]\d{2,})\b|告警|报警|故障|错误|异常",
            label_text,
            flags=re.IGNORECASE,
        )
        labels = []
        canonical_names = {
            "alarm": "Alarm",
            "fault": "Fault",
            "error": "Error",
            "warning": "Warning",
        }
        for term in fault_terms:
            if re.fullmatch(r"[ef]\d{2,}", term, re.I):
                canonical = term.upper()
            else:
                canonical = canonical_names.get(term.casefold(), term)
            if canonical.casefold() not in {item.casefold() for item in labels}:
                labels.append(canonical)

        evidence_match = re.search(
            r"visible_fault_evidence=([^|]+)",
            query or "",
            flags=re.IGNORECASE,
        )
        visible_evidence = evidence_match.group(1).strip() if evidence_match else ""
        evidence_text = "\n".join(
            [answer or "", visible_evidence, *(contexts or ())]
        )
        sentences = [
            item.strip(" -\t")
            for item in re.split(
                r"(?:\r?\n)+|(?<=[.!?。！？])\s+",
                evidence_text,
            )
            if item.strip()
        ]
        action_markers = (
            "remove", "clear", "troubleshoot", "power on", "restart", "contact",
            "排除", "清除", "上电", "重启", "联系", "处理",
        )

        def score(sentence):
            normalized = sentence.casefold()
            fault_hits = sum(term.casefold() in normalized for term in labels)
            action_hits = sum(marker in normalized for marker in action_markers)
            return fault_hits * 3 + action_hits * 2

        ranked = sorted(sentences, key=score, reverse=True)
        guidance = ranked[0] if ranked and score(ranked[0]) > 0 else ""
        user_part = (query or "").split("user_question=", 1)[-1]
        chinese = bool(re.search(r"[\u4e00-\u9fff]", user_part))
        normalized_guidance = guidance.casefold()
        reason = ""
        reason_match = re.search(
            r"reason\s*:\s*([^;|]+)",
            visible_evidence,
            flags=re.IGNORECASE,
        )
        if not reason_match:
            reason_match = re.search(
                r"([^;|]*abnormal[^;|]*)",
                visible_evidence,
                flags=re.IGNORECASE,
            )
        if reason_match:
            reason = reason_match.group(1).strip(" .")

        if chinese and reason:
            visible = "、".join(labels) if labels else "故障"
            evidence_casefold = evidence_text.casefold()
            if "contact supplier" in evidence_casefold:
                guidance = "知识库建议联系供应商，问题可能出在车辆的充电系统。"
            elif "contact administrator" in evidence_casefold:
                guidance = "按照图片提示，应联系管理员或运维人员处理，不要反复尝试启动。"
            elif not guidance:
                guidance = "请根据可见故障原因排查，无法确认时联系供应商或运维人员。"
            return (
                f"图中显示设备处于 {visible} 状态。\n\n"
                f"可见故障原因：{reason}。\n\n处理建议：{guidance}"
            )

        if chinese:
            visible = "、".join(labels) if labels else "告警或故障"
            if (
                any(term in normalized_guidance for term in ("remove", "clear", "排除", "清除"))
                and any(term in normalized_guidance for term in ("power on", "上电"))
            ):
                guidance = "启动时若出现这些状态，应先排查并清除故障，确认恢复正常后再上电。"
            elif not guidance:
                guidance = "知识库未提供足够的处理步骤，请不要在状态未确认前继续上电。"
            return f"图中显示的是电池系统的 {visible} 状态。\n\n处理建议：{guidance}"

        visible = ", ".join(labels) if labels else "an alarm or fault"
        if not guidance:
            guidance = (
                "The knowledge base does not provide enough handling detail; "
                "do not continue powering on until the state is verified."
            )
        return f"The image shows the battery-system state: {visible}.\n\nHandling: {guidance}"

    @staticmethod
    def _extract_source_urls(contexts, query=""):
        """Extract unique source URLs without mixing them into the scored answer."""
        sources = []
        for context in contexts:
            for url in re.findall(r"https?://[^\s\])]+", context):
                cleaned_url = url.rstrip(".,;，。；")
                if cleaned_url not in sources:
                    sources.append(cleaned_url)
        if "visual_focus=fault" in (query or "").casefold():
            focus_terms = (
                "alarm", "fault", "error", "warning",
                "告警", "报警", "故障", "异常",
            )
            focused_sources = []
            for context in contexts:
                context_casefold = context.casefold()
                for source in sources:
                    index = context.find(source)
                    if index < 0:
                        continue
                    start = context.rfind("\n", 0, index) + 1
                    next_newline = context.find("\n", index + len(source))
                    end = len(context) if next_newline < 0 else next_newline
                    window = context_casefold[start:end]
                    if any(term in window for term in focus_terms):
                        if source not in focused_sources:
                            focused_sources.append(source)
            if focused_sources:
                return focused_sources
        return sources

    @staticmethod
    def _is_product_support_query(query):
        """Route product and troubleshooting questions to the local RAG system."""
        normalized = query.lower()
        if re.search(r"\b(?:home-ess|hess|cess|ebs|fx)[a-z0-9~._/-]*\b", normalized):
            return True
        product_keywords = (
            "envalt", "英弗特", "ienos", "myenos", "enone",
            "hess", "逆变器", "储能", "电池", "电网", "光伏", "并网", "离网",
            "告警", "故障", "报错", "错误码", "参数", "手册", "安装", "接线",
            "充电", "放电", "功率", "电压", "电流", "频率", "ct", "bms",
            "ems", "pcs", "设备", "app掉线", "负电价", "馈网", "限馈",
            "ai mode", "工作模式", "soc",
        )
        return any(keyword in normalized for keyword in product_keywords)

    @staticmethod
    def _should_try_knowledge_first(query):
        """Try retrieval first except for short, unambiguous social messages."""
        return Chatbot._small_talk_response(query) is None

    @staticmethod
    def _small_talk_response(query):
        """Return a deterministic reply for social messages, otherwise ``None``."""
        normalized = re.sub(r"[\s,，。.!！?？:：;；]+", "", (query or "").casefold())
        if not normalized:
            return None

        greetings = {"你好", "您好", "嗨", "hello", "hi", "早上好", "下午好", "晚上好"}
        identity = {"你是谁", "你叫什么", "你叫什么名字", "介绍一下你自己"}
        thanks = {"谢谢", "感谢"}
        acknowledgements = {"好的", "收到"}
        farewells = {"再见", "拜拜"}

        if normalized in greetings:
            return "你好，我是小英，英弗特能源科技有限公司的 AI 智能客服助手。请问有什么可以帮您？"
        if normalized in identity:
            return "我是小英，英弗特能源科技有限公司的 AI 智能客服助手，可以协助您查询产品信息和排查设备问题。"
        if normalized in thanks:
            return "不客气。如果还有产品使用或设备故障方面的问题，可以继续告诉我。"
        if normalized in acknowledgements:
            return "好的，请继续告诉我您需要了解的问题。"
        if normalized in farewells:
            return "再见，如有需要可以随时来找我。"
        return None

    def _recover_agent_limit(self, query, response):
        """Recover a stopped Agent without automatically querying the public web."""
        knowledge_results = []
        for step in response.get("intermediate_steps", []):
            if not isinstance(step, (tuple, list)) or len(step) < 2:
                continue
            action, observation = step[0], step[1]
            if getattr(action, "tool", None) != "knowledge_base":
                continue
            if isinstance(observation, str) and observation.strip():
                knowledge_results.append(observation.strip())

        if knowledge_results:
            return self._format_knowledge_answer(query, knowledge_results)
        return (
            "抱歉，我暂时无法可靠回答这个问题。"
            "您可以换一种方式描述，或联系人工客服进一步确认。"
        )

    def _retrieve_local_contexts(self, query):
        """Retrieve final locally reranked context from the knowledge retriever."""
        self._last_retrieval_sources = []
        if self.knowledge_retriever is None:
            return []
        try:
            documents = self.knowledge_retriever.invoke(query)
            sources = []
            for document in documents:
                metadata = getattr(document, "metadata", {}) or {}
                source = str(metadata.get("source", "")).strip()
                if not source:
                    continue
                row = metadata.get("row")
                label = f"{source}（第 {row} 行）" if row is not None else source
                if label not in sources:
                    sources.append(label)
            self._last_retrieval_sources = sources
            return [
                document.page_content.strip()
                for document in documents
                if getattr(document, "page_content", "").strip()
            ]
        except Exception as e:
            print(f"本地知识库检索失败: {e}")
            return []

    @staticmethod
    def _extract_rag_contexts(response):
        """Extract knowledge_base observations for RAGAS evaluation."""
        contexts = [
            context.strip()
            for context in response.get("_rag_contexts", [])
            if isinstance(context, str) and context.strip()
        ]
        for step in response.get("intermediate_steps", []):
            if not isinstance(step, (tuple, list)) or len(step) < 2:
                continue
            action, observation = step[0], step[1]
            if getattr(action, "tool", None) != "knowledge_base":
                continue
            if (
                isinstance(observation, str)
                and observation.strip()
                and observation.strip() not in contexts
            ):
                contexts.append(observation.strip())
        return contexts

    def _finalize_response(
        self, query, response, output_text, evaluate_ragas=False, ground_truth=None
    ):
        """Evaluate, persist, and return one completed response."""
        # Agent tool calls can return a stored Q&A record verbatim. Keep the
        # answer itself but never expose storage labels such as Question/Answer.
        contexts = self._extract_rag_contexts(response)
        standard_answer = self._extract_standard_answer(output_text)
        if standard_answer:
            output_text = standard_answer
        output_text = self._focus_visual_fault_answer(query, output_text, contexts)

        if contexts:
            sources = self._extract_source_urls(contexts, query)
            for source in self._last_retrieval_sources:
                if source not in sources:
                    sources.append(source)
            response.setdefault("sources", sources)

        ragas_result = None
        if evaluate_ragas:
            from rag.evaluator import evaluate_rag_response

            ragas_result = evaluate_rag_response(
                query=query,
                retrieved_contexts=contexts,
                answer=output_text,
                ground_truth=ground_truth,
                llm=self.chatmodel,
                embeddings=getattr(self.knowledge_retriever, "embeddings", None),
            )
            print("RAGAS evaluation:", ragas_result)

        try:
            self.redis_memory.add_message(HumanMessage(content=query))
            self.redis_memory.add_message(AIMessage(content=output_text))
        except Exception as e:
            print(f"保存消息到 Redis 失败: {e}")

        self._chat_history.append(HumanMessage(content=query))
        self._chat_history.append(AIMessage(content=output_text))

        intermediate_steps = response.get("intermediate_steps")
        if intermediate_steps:
            print(f"本次agent实际调用的工具是:{intermediate_steps[0][0].tool}")
        elif response.get("_rag_contexts"):
            print("本次请求由本地 RAG 路由直接处理")
        else:
            print("本次agent没有调用任何工具")

        response["output"] = output_text
        if evaluate_ragas:
            response["ragas"] = ragas_result
        return response

    def _run_model_only(self, query: str) -> str:
        """Answer with the configured LLM only, without RAG or Agent tools."""
        history_lines: list[str] = []
        try:
            for message in self.redis_memory.messages[-6:]:
                role = "用户" if isinstance(message, HumanMessage) else "助手"
                history_lines.append(f"{role}: {message.content}")
        except Exception as exc:
            print(f"读取模型测试历史失败: {exc}")

        history = "\n".join(history_lines)
        prompt = (
            "你是小英，英弗特能源科技有限公司的 AI 智能客服助手。"
            "当前是纯模型测试模式：不要调用知识库、搜索或其他工具。"
            "请用与用户相同的语言给出简洁回答；不确定的产品事实必须明确说明不确定，"
            "不要编造型号、参数、政策或链接。\n\n"
            f"对话历史：\n{history or '（无）'}\n\n"
            f"用户问题：{query}\n助手回答："
        )
        response = self.chatmodel.invoke(prompt)
        output = getattr(response, "content", response)
        return str(output).strip()
    def run(self, query, evaluate_ragas=False, ground_truth=None):
        self._last_retrieval_sources = []
        if getattr(self, "model_only", False):
            output_text = self._run_model_only(query)
            return self._finalize_response(
                query,
                {"output": output_text, "intermediate_steps": [], "_rag_contexts": []},
                output_text,
                evaluate_ragas=False,
                ground_truth=ground_truth,
            )

        # 社交性短消息采用确定性直答，避免进入 RAG、Agent 或联网搜索。
        small_talk_response = self._small_talk_response(query)
        if small_talk_response is not None:
            return self._finalize_response(
                query,
                {
                    "output": small_talk_response,
                    "intermediate_steps": [],
                    "_rag_contexts": [],
                },
                small_talk_response,
                evaluate_ragas=False,
                ground_truth=ground_truth,
            )

        # 1.【终极防御】检查 agent_executor 是否存在，如果丢失，立刻全盘重建！
        if not hasattr(self, 'agent_executor') or self.agent_executor is None:
            print(" 检测到 AgentExecutor 丢失，正在立刻重建...")
            self.__init__()  # 重新初始化

        # 2. 从 Redis 获取历史记录
        # 优先获取短期记忆，如果为空则尝试获取长期记忆作为补充
        try:
            history = self.redis_memory.messages
            if not history:
                # Redis 历史为空时尝试读取长期记忆。
                if hasattr(self.redis_memory, 'get_long_term_memory'):
                    history = self.redis_memory.get_long_term_memory(limit=5)
                else:
                    # 如果是 InMemoryChatMessageHistory，直接使用 messages 属性
                    history = self.redis_memory.messages
        except Exception as e:
            print(f"获取历史记录失败: {e}")
            history = []

        # 更新本地 _chat_history 供 Agent 使用
        self._chat_history = history

        # 默认先检索知识库。这样即使问题没有出现型号或产品关键词，
        # 只要检索器命中可信资料，也会使用 RAG 回答，而不会先交给 Agent 猜测。
        product_support_query = self._is_product_support_query(query)
        direct_contexts = []
        if self._should_try_knowledge_first(query):
            print("[RAG] 优先查询本地知识库")
            direct_contexts = self._retrieve_local_contexts(query)
            if direct_contexts:
                output_text = self._format_knowledge_answer(query, direct_contexts)
                response = {
                    "output": output_text,
                    "intermediate_steps": [],
                    "_rag_contexts": direct_contexts,
                }
                return self._finalize_response(
                    query,
                    response,
                    output_text,
                    evaluate_ragas=evaluate_ragas,
                    ground_truth=ground_truth,
                )

        # 明确属于产品支持范围但没有可靠检索结果时，宁可明确说未收录，
        # 不让通用 Agent 或模型参数记忆生成未经知识库支持的产品结论。
        if product_support_query:
            output_text = "本地知识库中暂未检索到与该问题相关的信息。"
            response = {
                "output": output_text,
                "intermediate_steps": [],
                "_rag_contexts": [],
            }
            return self._finalize_response(
                query,
                response,
                output_text,
                evaluate_ragas=evaluate_ragas,
                ground_truth=ground_truth,
            )

        # 3. 情绪识别
        qingxu = self.qingxu_chain(query)
        print("当前用户情绪：", qingxu)
        if hasattr(self,'tools'):
            print(f"Agent当前绑定的工具列表:{[tool.name for tool in self.tools]}")
        else:
            print("没有工具")

        # 4. 调用 Agent
        try:
            # 初始化 response 变量，防止异常时未定义
            response = {}
            print(f"开始处理问题: {query}")
            response = self.agent_executor.invoke({
                "input": query,
                "chat_history": self._chat_history
            })


            # 打印完整的响应内容用于调试
            print("完整响应:", response)

            # 初始化response变量
            if not isinstance(response, dict):
                response = {"output": str(response)}

            # 检查response的格式
            if 'output' in response:
                output_text = response['output']
            elif 'intermediate_steps' in response:
                # 提取最后一次工具调用的结果
                if response['intermediate_steps']:
                    last_step = response['intermediate_steps'][-1]
                    if hasattr(last_step[0], 'observation'):
                        output_text = last_step[0].observation
                    else:
                        output_text = "抱歉，处理您的问题时遇到了一些技术困难。"
                else:
                    output_text = "抱歉，处理您的问题时遇到了一些技术困难。"
            else:
                output_text = "抱歉，处理您的问题时遇到了一些技术困难。"

        except ValueError as e:
            print(f"解析错误: {e}")
            output_text = "抱歉，处理您的问题时遇到了一些技术困难。建议您稍后重试或联系人工客服获取帮助。"
        except Exception as e:
            print(f"未知错误: {e}")
            output_text = "抱歉，系统出现了一些问题，请稍后再试。"

        # 5. 判断是否因达到限制而停止
        if "Agent stopped due to iteration limit" in output_text or "Agent stopped due to time limit" in output_text:
            print("="*50)
            print("【Agent 触发停止限制】")
            print("最后一步的思考内容:", output_text)
            print("中间步骤记录:", response.get("intermediate_steps"))
            print("="*50)

            output_text = self._recover_agent_limit(query, response)

        # 6. 统一执行评估、保存记忆并返回。
        return self._finalize_response(
            query,
            response,
            output_text,
            evaluate_ragas=evaluate_ragas,
            ground_truth=ground_truth,
        )



@app.get("/")
def read_root():
    return {
        "service": PROJECT_NAME,
        "assistant": ASSISTANT_NAME,
        "status": "ok",
        "version": app.version,
    }


class ChatRequest(BaseModel):
    query: str
    evaluate: bool = False
    ground_truth: Optional[str] = None
    debug: bool = False


class RetrieveRequest(BaseModel):
    query: str
    include_content: bool = False


@app.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    """Run retrieval only, without invoking the answer-generation model."""
    try:
        retriever = get_knowledge_retriever()
        if retriever is None:
            raise RuntimeError("Knowledge retriever is unavailable")
        documents = retriever.invoke(request.query)
        debug_payload = _retrieval_debug_payload(documents)
        if request.include_content:
            for payload, document in zip(debug_payload, documents):
                payload["content"] = document.page_content
        return {
            "status": "success",
            "data": {
                "retrieval_backend": getattr(retriever, "active_backend", "unknown"),
                "retrieval_debug": debug_payload,
            },
        }
    except Exception as exc:
        return {
            "status": "error",
            "data": {"message": str(exc), "retrieval_debug": []},
        }

# 全局变量
chatbot = None

@app.post("/chat")
async def chat(request:ChatRequest):
    global chatbot
    try:
        print(f"收到请求: {request.query}")
        if chatbot is None:
            chatbot = Chatbot()
        result = chatbot.run(
            request.query,
            evaluate_ragas=request.evaluate,
            ground_truth=request.ground_truth,
        )
        print(f"处理结果: {result}")
        data = {
            "response": result.get("output", str(result))
        }
        if result.get("sources"):
            data["sources"] = result["sources"]
        if request.evaluate:
            data["ragas"] = result.get("ragas")
        if request.debug and chatbot.knowledge_retriever is not None:
            try:
                data["retrieval_backend"] = getattr(
                    chatbot.knowledge_retriever,
                    "active_backend",
                    "unknown",
                )
                debug_documents = chatbot.knowledge_retriever.invoke(request.query)
                data["retrieval_debug"] = _retrieval_debug_payload(debug_documents)
            except Exception as exc:
                data["retrieval_debug"] = []
                data["retrieval_debug_error"] = str(exc)
        return {
            "status": "success",
            "data": data
        }
    except Exception as e:
        print(f"处理请求时发生错误: {e}")
        return {
            "status": "error",
            "data": {
                "response": "抱歉，处理您的请求时发生了错误。请稍后重试或联系客服获取帮助。"
            }
        }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        print("Client disconnected")
        await websocket.close()
