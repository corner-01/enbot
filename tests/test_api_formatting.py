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


if __name__ == "__main__":
    unittest.main()
