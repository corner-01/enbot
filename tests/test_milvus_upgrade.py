import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from rag.hybrid_search import HybridSearchEngine
from rag.knowledge_retriever import load_knowledge_chunks
from rag.vector_store import VectorStoreSettings


class DenseRetriever:
    def __init__(self, documents):
        self.documents = documents

    def invoke(self, _query):
        return list(self.documents)


class VectorStoreWithoutChromaGet:
    """Milvus-like object: deliberately has no Chroma ``get`` method."""


class MilvusUpgradeTests(unittest.TestCase):
    def _engine(self, documents, final_k=2):
        with patch.dict(os.environ, {"RERANKER_ENABLED": "false"}):
            return HybridSearchEngine(
                vectorstore=VectorStoreWithoutChromaGet(),
                dense_retriever=DenseRetriever(documents),
                documents=documents,
                dense_k=len(documents),
                lexical_k=len(documents),
                rerank_k=len(documents),
                final_k=final_k,
            )

    def test_json_qa_sources_are_loaded_and_duplicates_are_skipped(self):
        rows = [
            {
                "instruction": "What does SoC mean?",
                "input": "",
                "output": "SoC means State of Charge.",
            },
            {
                "instruction": "What does SoC mean?",
                "input": "",
                "output": "SoC means State of Charge.",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "qa.json"
            source.write_text(json.dumps(rows), encoding="utf-8")
            with patch.dict(
                os.environ,
                {"RAG_CHUNK_SIZE": "1000", "RAG_CHUNK_OVERLAP": "0"},
            ):
                documents = load_knowledge_chunks(directory)

        self.assertEqual(len(documents), 1)
        self.assertIn("Question: What does SoC mean?", documents[0].page_content)
        self.assertIn("Answer: SoC means State of Charge.", documents[0].page_content)
        self.assertEqual(documents[0].metadata["format"], "instruction_input_output")

    def test_hybrid_search_accepts_source_documents_without_chroma_get(self):
        documents = [
            Document(
                page_content="Question: HESS-HY-T-10K voltage Answer: 230 V",
                metadata={"source": "qa.xlsx"},
            ),
            Document(
                page_content="Question: battery SOC Answer: state of charge",
                metadata={"source": "faq.txt"},
            ),
        ]
        with patch.dict(os.environ, {"RERANKER_ENABLED": "false"}):
            engine = HybridSearchEngine(
                vectorstore=VectorStoreWithoutChromaGet(),
                dense_retriever=DenseRetriever(documents),
                documents=documents,
                dense_k=2,
                lexical_k=2,
                rerank_k=2,
                final_k=1,
            )
            results = engine.search("HESS-HY-T-10K voltage")

        self.assertEqual(len(results), 1)
        self.assertIn("230 V", results[0].page_content)

    def test_out_of_domain_query_is_rejected(self):
        documents = [
            Document(
                page_content="Question: battery SOC Answer: state of charge",
                metadata={"source": "faq.json"},
            )
        ]
        engine = self._engine(documents)
        self.assertEqual(engine.search("如何制作巧克力蛋糕？"), [])

    def test_fault_code_must_match_exactly(self):
        documents = [
            Document(
                page_content=(
                    "Question: HOME-ESS-LV-3.68K E.19故障码\n"
                    "Answer: E.19表示反接保护。"
                ),
                metadata={"source": "alarms.json", "row": 58},
            ),
            Document(
                page_content=(
                    "Question: HOME-ESS-LV-3.68K E.20故障码\n"
                    "Answer: E.20表示充电高温保护。"
                ),
                metadata={"source": "alarms.json", "row": 1},
            ),
        ]
        results = self._engine(documents, final_k=1).search(
            "HOME-ESS-LV-3.68K的E.20故障码是什么意思？"
        )
        self.assertEqual(len(results), 1)
        self.assertIn("E.20表示充电高温保护", results[0].page_content)

    def test_unknown_exact_model_is_rejected(self):
        documents = [
            Document(
                page_content="Question: HESS-HY-S-6K-S grid data Answer: voltage",
                metadata={"source": "manual.json"},
            )
        ]
        engine = self._engine(documents)
        self.assertEqual(
            engine.search("CESS-HY-999K的LCD电网数据NO.1到NO.7是什么？"),
            [],
        )

    def test_environment_selects_milvus_and_collection(self):
        with patch.dict(
            os.environ,
            {
                "VECTOR_STORE": "milvus",
                "VECTOR_STORE_FALLBACK": "chroma",
                "MILVUS_URI": "http://127.0.0.1:19530",
                "MILVUS_COLLECTION": "enbot_test",
            },
            clear=False,
        ):
            settings = VectorStoreSettings.from_env()

        self.assertEqual(settings.backend, "milvus")
        self.assertEqual(settings.fallback_backend, "chroma")
        self.assertEqual(settings.milvus_collection, "enbot_test")


if __name__ == "__main__":
    unittest.main()
