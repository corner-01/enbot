import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from rag.hybrid_search import HybridSearchEngine
from rag.knowledge_retriever import _load_knowledge_corpus, load_knowledge_chunks
from rag.vector_store import VectorStoreSettings


class DenseRetriever:
    def __init__(self, documents):
        self.documents = documents

    def invoke(self, _query):
        return list(self.documents)


class MilvusLikeVectorStore:
    """Minimal vector store without a collection-wide ``get`` method."""


class MilvusUpgradeTests(unittest.TestCase):
    def _engine(self, documents, final_k=2):
        with patch.dict(os.environ, {"RERANKER_ENABLED": "false"}):
            return HybridSearchEngine(
                vectorstore=MilvusLikeVectorStore(),
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

    def test_duplicate_sources_keep_parent_aliases(self):
        content = "Question: What is battery SoC?\nAnswer: State of charge."
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "first.txt").write_text(content, encoding="utf-8")
            Path(directory, "duplicate.txt").write_text(content, encoding="utf-8")
            chunks, parents = _load_knowledge_corpus(directory)

        self.assertEqual(len(chunks), 1)
        self.assertIn("first.txt::document", parents)
        self.assertIn("duplicate.txt::document", parents)

    def test_long_qa_is_chunked_for_search_but_kept_as_complete_parent(self):
        long_answer = "first-section " + ("battery detail " * 30) + "final-section"
        rows = [
            {
                "instruction": "How should the HOME-ESS-LV-3.2K be maintained?",
                "input": "",
                "output": long_answer,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "long-qa.json"
            source.write_text(json.dumps(rows), encoding="utf-8")
            with patch.dict(
                os.environ,
                {"RAG_CHUNK_SIZE": "120", "RAG_CHUNK_OVERLAP": "20"},
            ):
                chunks, parents = _load_knowledge_corpus(directory)

        self.assertGreater(len(chunks), 1)
        record_ids = {document.metadata["record_id"] for document in chunks}
        self.assertEqual(record_ids, {"long-qa.json::row:1"})
        self.assertEqual(
            parents["long-qa.json::row:1"].page_content,
            f"Question: How should the HOME-ESS-LV-3.2K be maintained?\n"
            f"Answer: {long_answer}",
        )

    def test_hybrid_search_restores_full_parent_after_child_ranking(self):
        record_id = "manual.json::row:7"
        full_answer = "first-section " + ("battery detail " * 30) + "final-section"
        parent = Document(
            page_content=(
                "Question: How should the HOME-ESS-LV-3.2K be maintained?\n"
                f"Answer: {full_answer}"
            ),
            metadata={
                "source": "manual.json",
                "row": 7,
                "format": "instruction_input_output",
                "record_id": record_id,
            },
        )
        child = Document(
            page_content=(
                "Question: How should the HOME-ESS-LV-3.2K be maintained?\n"
                "Answer: first-section battery detail"
            ),
            metadata={
                "source": "manual.json",
                "row": 7,
                "format": "instruction_input_output",
                "record_id": record_id,
                "start_index": 0,
            },
        )
        with patch.dict(os.environ, {"RERANKER_ENABLED": "false"}):
            engine = HybridSearchEngine(
                vectorstore=MilvusLikeVectorStore(),
                dense_retriever=DenseRetriever([child]),
                documents=[child],
                parent_documents={record_id: parent},
                dense_k=1,
                lexical_k=1,
                rerank_k=1,
                final_k=1,
            )
            results = engine.search("HOME-ESS-LV-3.2K battery maintenance")

        self.assertEqual(len(results), 1)
        self.assertIn("final-section", results[0].page_content)
        self.assertTrue(results[0].metadata["parent_restored"])
        self.assertEqual(results[0].metadata["matched_start_index"], 0)

    def test_parent_restore_supports_chunks_from_existing_milvus_index(self):
        record_id = "faq.json::row:9"
        parent = Document(
            page_content="Question: What is battery SoC?\nAnswer: Complete answer.",
            metadata={"source": "faq.json", "row": 9, "record_id": record_id},
        )
        old_index_child = Document(
            page_content="Question: What is battery SoC? Answer: Partial answer.",
            metadata={"source": "faq.json", "row": 9},
        )
        engine = self._engine([])
        engine.parent_documents = {record_id: parent}

        restored = engine._restore_parent_documents([old_index_child], limit=1)

        self.assertEqual(restored[0].page_content, parent.page_content)

    def test_lexical_metadata_enriches_old_dense_candidate(self):
        record_id = "manual.json::row:7"
        content = "Question: HOME-ESS-LV-3.2K maintenance\nAnswer: first section"
        old_dense_child = Document(
            page_content=content,
            metadata={"source": "manual.json"},
        )
        current_child = Document(
            page_content=content,
            metadata={
                "source": "manual.json",
                "row": 7,
                "record_id": record_id,
                "start_index": 0,
            },
        )
        parent = Document(
            page_content=content + " and complete final section",
            metadata={"source": "manual.json", "row": 7, "record_id": record_id},
        )
        with patch.dict(os.environ, {"RERANKER_ENABLED": "false"}):
            engine = HybridSearchEngine(
                vectorstore=MilvusLikeVectorStore(),
                dense_retriever=DenseRetriever([old_dense_child]),
                documents=[current_child],
                parent_documents={record_id: parent},
                dense_k=1,
                lexical_k=1,
                rerank_k=1,
                final_k=1,
            )
            results = engine.search("HOME-ESS-LV-3.2K battery maintenance")

        self.assertEqual(results[0].page_content, parent.page_content)
        self.assertTrue(results[0].metadata["parent_restored"])

    def test_document_without_parent_is_returned_unchanged(self):
        document = Document(
            page_content="Question: battery SoC Answer: state of charge",
            metadata={"source": "faq.json"},
        )
        engine = self._engine([document])

        results = engine.search("battery SoC")

        self.assertEqual(results[0].page_content, document.page_content)
        self.assertNotIn("parent_restored", results[0].metadata)

    def test_oversized_markdown_parent_restores_only_requested_section(self):
        record_id = "web-manual.json::row:2"
        parent = Document(
            page_content=(
                "Question: [WEB side]Instructions\nAnswer: "
                "## [WEB side]Instructions->Basic Functions\n"
                + ("basic details " * 30)
                + "\n### [WEB side]Instructions->Monitoring->Home page\n"
                "The home page shows installed capacity and statistical data.\n"
                "| A | Navigation bar |\n"
                "![](https://example.com/home.png)\n"
                "##### [WEB side]Instructions->Monitoring->Home page->Add Device\n"
                "Nested subsection must not replace the exact requested heading.\n"
                "### [WEB side]Instructions->Monitoring->Station page\n"
                "Station-only details must not be returned."
            ),
            metadata={
                "source": "web-manual.json",
                "row": 2,
                "record_id": record_id,
            },
        )
        child = Document(
            page_content=(
                "### [WEB side]Instructions->Monitoring->Home page\n"
                "The home page shows installed capacity and statistical data."
            ),
            metadata={
                "source": "web-manual.json",
                "row": 2,
                "record_id": record_id,
                "start_index": 450,
            },
        )
        with patch.dict(os.environ, {"RERANKER_ENABLED": "false"}):
            engine = HybridSearchEngine(
                vectorstore=MilvusLikeVectorStore(),
                dense_retriever=DenseRetriever([child]),
                documents=[child],
                parent_documents={record_id: parent},
                dense_k=1,
                lexical_k=1,
                rerank_k=1,
                final_k=1,
                parent_full_max_chars=200,
            )
            results = engine.search(
                "[WEB side]Instructions->Monitoring->Home page"
            )

        self.assertEqual(len(results), 1)
        self.assertIn("installed capacity", results[0].page_content)
        self.assertIn("https://example.com/home.png", results[0].page_content)
        self.assertNotIn("Nested subsection", results[0].page_content)
        self.assertNotIn("Station-only details", results[0].page_content)
        self.assertTrue(results[0].metadata["parent_section_extracted"])
        self.assertEqual(results[0].metadata["parent_total_chars"], len(parent.page_content))

    def test_hybrid_search_accepts_source_documents_without_collection_get(self):
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
                vectorstore=MilvusLikeVectorStore(),
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

    def test_optuna_configuration_updates_values_and_clears_cache(self):
        documents = [
            Document(
                page_content="Question: battery SOC Answer: state of charge",
                metadata={"source": "faq.json", "row": 1},
            )
        ]
        engine = self._engine(documents)
        engine._set_cached("battery SOC", documents)
        replacement_dense = DenseRetriever(documents)

        engine.configure(
            dense_retriever=replacement_dense,
            dense_k=30,
            lexical_k=40,
            rerank_k=8,
            final_k=2,
            rrf_k=40.0,
            dense_weight=1.25,
            lexical_weight=0.75,
            rerank_relative_threshold=0.7,
            rerank_absolute_threshold=0.05,
            rerank_min_score=0.02,
        )

        self.assertIs(engine.dense_retriever, replacement_dense)
        self.assertEqual(engine.dense_k, 30)
        self.assertEqual(engine.lexical_k, 40)
        self.assertEqual(engine.rerank_k, 8)
        self.assertEqual(engine.rrf_k, 40.0)
        self.assertEqual(engine.dense_weight, 1.25)
        self.assertEqual(engine.lexical_weight, 0.75)
        self.assertEqual(engine.rerank_min_score, 0.02)
        self.assertIsNone(engine._get_cached("battery SOC"))

    def test_near_equal_reranker_scores_use_rrf_as_tie_breaker(self):
        engine = self._engine([])
        weaker_rrf = Document(
            page_content="system parameters",
            metadata={"reranker_score": 0.9999786, "rrf_score": 0.0238},
        )
        stronger_rrf = Document(
            page_content="module parameters",
            metadata={"reranker_score": 0.9999780, "rrf_score": 0.0246},
        )
        ranked = sorted(
            [weaker_rrf, stronger_rrf],
            key=engine._rerank_tie_key,
            reverse=True,
        )
        self.assertEqual(ranked[0].page_content, "module parameters")

    def test_document_download_queries_are_asset_requests(self):
        engine = self._engine([])
        self.assertTrue(
            engine._is_asset_request("CESS-HV-9.4K Quick Installation Guide")
        )

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
                "MILVUS_URI": "http://127.0.0.1:19530",
                "MILVUS_COLLECTION": "enbot_test",
            },
            clear=False,
        ):
            settings = VectorStoreSettings.from_env()

        self.assertEqual(settings.backend, "milvus")
        self.assertEqual(settings.milvus_collection, "enbot_test")


if __name__ == "__main__":
    unittest.main()
