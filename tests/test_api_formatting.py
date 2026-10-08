import unittest

from app.api import Chatbot


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


if __name__ == "__main__":
    unittest.main()
