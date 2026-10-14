import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evaluation.evaluate_new_data_full import (
    build_cases,
    first_matching_rank,
    normalize_source,
    summarize,
)


class FullRetrievalEvaluationTests(unittest.TestCase):
    def test_build_cases_keeps_all_valid_rows_and_accepts_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "一类").mkdir()
            (root / "二类").mkdir()
            pd.DataFrame(
                [
                    {"问题": "EnValt 是什么？", "答案": "品牌说明"},
                    {"问题": "缺少答案", "答案": ""},
                ]
            ).to_excel(root / "一类" / "a.xlsx", index=False)
            pd.DataFrame(
                [
                    {"question": "EnValt 是什么？", "answer": "品牌说明"},
                    {"question": "电池寿命？", "answer": "6000 次循环"},
                ]
            ).to_excel(root / "二类" / "b.xlsx", index=False)

            cases, manifest = build_cases(root)

            self.assertEqual(len(cases), 3)
            duplicate_cases = [case for case in cases if case["query"] == "EnValt 是什么？"]
            self.assertEqual(len(duplicate_cases), 2)
            self.assertEqual(duplicate_cases[0]["reference"], "品牌说明")
            self.assertEqual(len(duplicate_cases[0]["acceptable_documents"]), 2)
            self.assertEqual(manifest["skipped"]["empty_answer"], 1)

    def test_source_separator_and_row_are_normalized(self):
        documents = [{"source": "售前/市场.xlsx", "row": "7"}]
        accepted = [{"source": r"售前\市场.xlsx", "row": 7}]
        self.assertEqual(first_matching_rank(documents, accepted), 1)
        self.assertEqual(normalize_source(r"售前\市场.xlsx"), "售前/市场.xlsx")

    def test_summary_calculates_strict_and_equivalent_metrics(self):
        cases = [
            {"name": "A", "expected_source": "a.xlsx"},
            {"name": "B", "expected_source": "b.xlsx"},
        ]
        results = [
            {
                "name": "A",
                "strict_rank": 1,
                "equivalent_rank": 1,
                "duration_seconds": 1.0,
                "error": None,
            },
            {
                "name": "B",
                "strict_rank": 0,
                "equivalent_rank": 2,
                "duration_seconds": 3.0,
                "error": None,
            },
        ]

        report = summarize(cases, results)

        self.assertEqual(report["strict"]["recall_at_2"], 0.5)
        self.assertEqual(report["equivalent"]["recall_at_2"], 1.0)
        self.assertEqual(report["equivalent"]["mrr"], 0.75)
        self.assertEqual(report["latency_seconds"]["p50"], 2.0)


if __name__ == "__main__":
    unittest.main()