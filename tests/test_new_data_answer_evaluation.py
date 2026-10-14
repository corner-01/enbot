import unittest

from evaluation.evaluate_new_data_answers import summarize


class NewDataAnswerEvaluationTests(unittest.TestCase):
    def test_summary_reports_semantic_accuracy_and_errors_separately(self):
        cases = [
            {"name": "A"},
            {"name": "B"},
            {"name": "C"},
        ]
        deterministic = {
            "token_f1": 0.5,
            "char_similarity": 0.4,
            "number_unit_recall": None,
            "model_code_recall": None,
            "url_recall": None,
        }
        results = [
            {
                "name": "A",
                "expected_source": "a.xlsx",
                "answer": "ok",
                "deterministic": deterministic,
                "generation_seconds": 1.0,
                "judge_seconds": 0.5,
                "judge": {
                    "correctness": 5,
                    "completeness": 4,
                    "relevance": 5,
                    "hallucination": False,
                },
                "semantic_pass": True,
            },
            {
                "name": "B",
                "expected_source": "a.xlsx",
                "answer": "wrong",
                "deterministic": deterministic,
                "generation_seconds": 2.0,
                "judge_seconds": 0.5,
                "judge": {
                    "correctness": 2,
                    "completeness": 2,
                    "relevance": 4,
                    "hallucination": True,
                },
                "semantic_pass": False,
            },
            {
                "name": "C",
                "expected_source": "b.xlsx",
                "generation_seconds": 3.0,
                "chat_error": "timeout",
            },
        ]

        report = summarize(cases, results)

        self.assertEqual(report["semantic_accuracy"], 0.5)
        self.assertEqual(report["completion_adjusted_accuracy"], 1 / 3)
        self.assertEqual(report["chat_errors"], 1)
        self.assertEqual(report["hallucination_rate"], 0.5)
        self.assertEqual(report["per_source"]["a.xlsx"]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()