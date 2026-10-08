import unittest

from integrations.feishu_bot import (
    extract_enbot_result,
    feedback_ack_message,
    feedback_prompt,
    parse_feedback_command,
)


class FeishuFeedbackTests(unittest.TestCase):
    def test_extract_result_keeps_request_id(self):
        result = extract_enbot_result(
            {
                "status": "success",
                "request_id": "request-123",
                "data": {"response": "answer"},
            }
        )
        self.assertEqual(result.answer, "answer")
        self.assertEqual(result.request_id, "request-123")

    def test_parse_positive_and_negative_feedback(self):
        self.assertEqual(parse_feedback_command("有帮助").rating, "up")
        negative = parse_feedback_command("没帮助：缺少操作步骤")
        self.assertEqual(negative.rating, "down")
        self.assertEqual(negative.comment, "缺少操作步骤")
        self.assertIsNone(parse_feedback_command("这个功能有帮助吗？"))

    def test_user_messages_are_readable(self):
        self.assertIn("有帮助", feedback_prompt())
        self.assertIn("没帮助", feedback_prompt())
        self.assertIn("人工复核", feedback_ack_message("down"))


if __name__ == "__main__":
    unittest.main()
