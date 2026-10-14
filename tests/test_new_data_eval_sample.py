import json
import tempfile
import unittest
from pathlib import Path

from evaluation.build_new_data_eval_sample import balanced_sample
from evaluation.evaluate_new_data_answers import load_answer_cases


class NewDataEvalSampleTests(unittest.TestCase):
    def test_balanced_sample_is_deterministic_and_covers_sources(self):
        cases = []
        for source in ("a.xlsx", "b.xlsx", "c.xlsx"):
            for row in range(1, 6):
                cases.append(
                    {
                        "name": f"{source}-{row}",
                        "query": f"{source} question {row}",
                        "reference": f"answer {row}",
                        "expected_source": source,
                        "expected_row": row,
                    }
                )
        cases.append(dict(cases[0]))

        first, stats = balanced_sample(cases, 9, seed=42)
        second, _ = balanced_sample(cases, 9, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(stats["covered_sources"], 3)
        self.assertEqual(stats["duplicates_removed"], 1)
        self.assertEqual({case["name"] for case in first}, {f"RAG100-{i:03d}" for i in range(1, 10)})

    def test_load_answer_cases_accepts_manifest_object(self):
        payload = {
            "seed": 42,
            "cases": [
                {
                    "name": "RAG100-001",
                    "query": "q",
                    "reference": "a",
                    "expected_source": "a.xlsx",
                    "expected_row": 2,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            cases, manifest = load_answer_cases(path)

        self.assertEqual(len(cases), 1)
        self.assertEqual(manifest["seed"], 42)
        self.assertEqual(manifest["case_file"], str(path.resolve()))


if __name__ == "__main__":
    unittest.main()