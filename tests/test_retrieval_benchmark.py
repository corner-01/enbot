import unittest

from evaluation.retrieval_benchmark import first_relevant_rank


class RetrievalBenchmarkTests(unittest.TestCase):
    def test_source_and_row_identify_the_exact_qa_record(self):
        case = {"expected_source": "faq.json", "expected_row": 7}
        documents = [
            {"source": "faq.json", "row": 3, "preview": "wrong row"},
            {"source": "faq.json", "row": 7, "preview": "correct row"},
        ]
        self.assertEqual(first_relevant_rank(case, documents), 2)

    def test_missing_expected_row_is_not_relevant(self):
        case = {"expected_source": "faq.json", "expected_row": 7}
        documents = [
            {"source": "faq.json", "row": 3, "preview": "wrong row"},
        ]
        self.assertEqual(first_relevant_rank(case, documents), 0)


if __name__ == "__main__":
    unittest.main()
