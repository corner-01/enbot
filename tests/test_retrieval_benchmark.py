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

    def test_equivalent_document_is_accepted(self):
        case = {
            "expected_source": "raw.json",
            "expected_row": 2,
            "acceptable_documents": [
                {"source": "raw.json", "row": 2},
                {"source": "consolidated.json", "row": 7},
            ],
        }
        documents = [
            {
                "source": "consolidated.json",
                "row": 7,
                "preview": "same answer in a consolidated record",
            }
        ]
        self.assertEqual(first_relevant_rank(case, documents), 1)


if __name__ == "__main__":
    unittest.main()
