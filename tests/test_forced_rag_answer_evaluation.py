import json
import unittest
from unittest.mock import patch

from evaluation.evaluate_forced_rag_answers import retrieve_direct


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class ForcedRagAnswerEvaluationTests(unittest.TestCase):
    def test_retrieve_direct_reads_b_node_documents(self):
        payload = {
            "status": "success",
            "data": {
                "elapsed_seconds": 0.25,
                "documents": [
                    {
                        "content": "Question: q\nAnswer: a",
                        "metadata": {"source": "x.xlsx", "row": 2},
                    }
                ],
            },
        }
        with patch(
            "evaluation.evaluate_forced_rag_answers.urllib.request.urlopen",
            return_value=_FakeResponse(payload),
        ):
            documents, elapsed = retrieve_direct(
                "http://example.test/v1/retrieve",
                api_key="secret",
                query="q",
                timeout=10,
            )

        self.assertEqual(elapsed, 0.25)
        self.assertEqual(documents[0]["metadata"]["row"], 2)


if __name__ == "__main__":
    unittest.main()