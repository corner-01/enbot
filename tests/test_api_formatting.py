import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.api import (
    Chatbot,
    _web_search_enabled,
    get_active_model_name,
    get_llm_provider,
)


class ApiFormattingTests(unittest.TestCase):
    def test_standard_answer_removes_image_shell_and_markdown_escapes(self):
        context = (
            "Question: [WEB side]Instructions->Monitoring->Home page\n"
            "Answer: The home page shows installed capacity.\n"
            r"![]\(https://example.com/home.png\)" + "\n"
            r'| \*\*\<font style="color:#FFFFFF;"\>Label\</font\>\*\* |'
        )

        answer = Chatbot._extract_standard_answer(context)

        self.assertNotIn("https://", answer)
        self.assertNotIn("![]()", answer)
        self.assertNotIn("<font", answer)
        self.assertNotIn(r"\*", answer)
        self.assertIn("| **Label** |", answer)

    def test_visual_fault_answer_keeps_only_state_and_handling(self):
        query = (
            "visual_focus=fault | battery system display shows Alarm Fault | "
            "user_question=这张图有什么故障？"
        )
        broad_answer = (
            "主页可查看天气、电压、SOC 和系统状态。\n"
            "If the display indicates an Alarm or a Fault, you need to remove "
            "the fault and then power on.\n"
            "The system also provides charging-mode settings."
        )

        answer = Chatbot._focus_visual_fault_answer(query, broad_answer)

        self.assertIn("Alarm", answer)
        self.assertIn("Fault", answer)
        self.assertIn("清除故障", answer)
        self.assertNotIn("天气", answer)
        self.assertNotIn("charging-mode", answer)

    def test_visual_fault_sources_keep_fault_adjacent_image_only(self):
        query = "visual_focus=fault | battery display Alarm Fault"
        context = (
            "Normal state https://example.test/normal.png\n"
            "If the display indicates Alarm or Fault, remove the fault first. "
            "https://example.test/alarm.png\n"
            "Power switch https://example.test/switch.png"
        )

        sources = Chatbot._extract_source_urls([context], query)

        self.assertEqual(sources, ["https://example.test/alarm.png"])

    def test_visual_fault_answer_preserves_specific_ev_reason(self):
        query = (
            "visual_focus=fault | device display shows Fault | "
            "visible_fault_evidence=Device faulty. Please contact administrator.; "
            "Reason: Vehicle CP abnormal negative voltage | "
            "user_question=图中是什么故障？"
        )
        context = (
            "Abnormal signal from the vehicle detected. "
            "Action: Contact supplier. The issue may be with the vehicle's "
            "charging system."
        )

        answer = Chatbot._focus_visual_fault_answer(query, "Fault", [context])

        self.assertIn("Vehicle CP abnormal negative voltage", answer)
        self.assertIn("联系供应商", answer)
        self.assertNotIn("电池系统", answer)

    def test_company_platform_and_model_queries_route_to_rag(self):
        queries = (
            "EnValt是什么？",
            "无锡英弗特的主营业务是什么？",
            "iEnOS是什么平台？",
            "myEnOS面向哪些用户？",
            "EnOne是什么？",
            "EBS-H-16K是什么？",
            "FX-10K-T怎么设置？",
        )

        for query in queries:
            with self.subTest(query=query):
                self.assertTrue(Chatbot._is_product_support_query(query))

    def test_general_chat_does_not_force_product_rag(self):
        for query in ("你好", "你是谁", "Python是什么？"):
            with self.subTest(query=query):
                self.assertFalse(Chatbot._is_product_support_query(query))

    def test_knowledge_first_skips_only_small_talk(self):
        self.assertFalse(Chatbot._should_try_knowledge_first("你好！"))
        self.assertFalse(Chatbot._should_try_knowledge_first("你是谁？"))
        self.assertTrue(
            Chatbot._should_try_knowledge_first(
                "Which personnel may remove or dismantle safety facilities?"
            )
        )
        self.assertTrue(Chatbot._should_try_knowledge_first("Python是什么？"))

    def test_small_talk_returns_without_rag_agent_or_web_search(self):
        bot = Chatbot.__new__(Chatbot)
        bot.model_only = False
        bot.agent_executor = Mock()
        bot.redis_memory = SimpleNamespace(messages=[])
        bot._retrieve_local_contexts = Mock()
        bot._finalize_response = Mock(
            side_effect=lambda _query, _response, output, **_kwargs: {"output": output}
        )

        with patch("app.api.search") as web_search:
            result = bot.run("你好！")

        self.assertIn("我是小英", result["output"])
        bot._retrieve_local_contexts.assert_not_called()
        bot.agent_executor.invoke.assert_not_called()
        web_search.assert_not_called()

    def test_agent_limit_recovery_does_not_force_web_search(self):
        bot = Chatbot.__new__(Chatbot)
        bot._format_knowledge_answer = Mock()

        with patch("app.api.search") as web_search:
            answer = bot._recover_agent_limit(
                "一个没有可靠答案的问题",
                {"intermediate_steps": []},
            )

        self.assertIn("无法可靠回答", answer)
        bot._format_knowledge_answer.assert_not_called()
        web_search.assert_not_called()

    def test_web_search_is_disabled_by_default_and_can_be_opted_in(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_web_search_enabled())
        with patch.dict(os.environ, {"ENBOT_WEB_SEARCH_ENABLED": "true"}, clear=True):
            self.assertTrue(_web_search_enabled())

    def test_unknown_in_domain_wording_uses_retrieval_hit_before_agent(self):
        bot = Chatbot.__new__(Chatbot)
        bot.agent_executor = Mock()
        bot.redis_memory = SimpleNamespace(messages=[])
        bot._retrieve_local_contexts = Mock(return_value=["trusted knowledge"])
        bot._format_knowledge_answer = Mock(return_value="grounded answer")
        bot._finalize_response = Mock(return_value={"output": "grounded answer"})

        result = bot.run("Which personnel may remove or dismantle safety facilities?")

        self.assertEqual(result["output"], "grounded answer")
        bot._retrieve_local_contexts.assert_called_once()
        bot.agent_executor.invoke.assert_not_called()

    def test_product_query_without_retrieval_hit_does_not_fall_back_to_agent(self):
        bot = Chatbot.__new__(Chatbot)
        bot.agent_executor = Mock()
        bot.redis_memory = SimpleNamespace(messages=[])
        bot._retrieve_local_contexts = Mock(return_value=[])
        bot._finalize_response = Mock(
            side_effect=lambda _query, _response, output, **_kwargs: {"output": output}
        )

        result = bot.run("FX-10K-T 的未知参数是什么？")

        self.assertIn("暂未检索到", result["output"])
        bot.agent_executor.invoke.assert_not_called()

    def test_llm_provider_defaults_to_ollama_and_selects_openai_model(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": ""}, clear=False):
            self.assertEqual(get_llm_provider(), "ollama")

        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "openai", "OPENAI_MODEL": "gpt-test"},
            clear=False,
        ):
            self.assertEqual(get_llm_provider(), "openai")
            self.assertEqual(get_active_model_name(), "gpt-test")

    def test_llm_provider_rejects_unknown_value(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "unsupported"}, clear=False):
            with self.assertRaises(ValueError):
                get_llm_provider()

if __name__ == "__main__":
    unittest.main()
