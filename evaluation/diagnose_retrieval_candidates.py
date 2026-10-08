"""Inspect expanded reranker candidates for selected development cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.knowledge_retriever import get_knowledge_retriever


DEFAULT_CASES = Path(__file__).with_name("retrieval_cases_v3.json")
DEFAULT_NAMES = ["DEV3-POS-048", "DEV3-POS-058"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--name", action="append", dest="names")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    if "holdout" in {part.lower() for part in args.cases.resolve().parts}:
        raise ValueError("诊断程序禁止读取holdout验证集")
    cases = json.loads(args.cases.read_text(encoding="utf-8-sig"))
    wanted = set(args.names or DEFAULT_NAMES)
    selected = [case for case in cases if case.get("name") in wanted]
    missing = wanted - {case["name"] for case in selected}
    if missing:
        raise ValueError(f"找不到案例: {sorted(missing)}")

    load_dotenv(PROJECT_ROOT / ".env")
    retriever = get_knowledge_retriever(raise_on_error=True)
    if retriever is None or retriever.hybrid_engine is None:
        raise RuntimeError("混合检索器初始化失败")
    engine = retriever.hybrid_engine
    dense_retriever = retriever.vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": engine.dense_k},
    )
    engine.configure(
        dense_retriever=dense_retriever,
        dense_k=engine.dense_k,
        lexical_k=engine.lexical_k,
        rerank_k=max(engine.rerank_k, 20),
        final_k=max(args.limit, 10),
        rrf_k=engine.rrf_k,
        dense_weight=engine.dense_weight,
        lexical_weight=engine.lexical_weight,
        rerank_relative_threshold=0.0,
        rerank_absolute_threshold=0.0,
        rerank_min_score=-1.0,
    )

    for case in selected:
        print("=" * 88)
        print(f"CASE: {case['name']}")
        print(f"QUERY: {case['query']}")
        print(
            "EXPECTED: "
            f"{case.get('expected_source')} row={case.get('expected_row')}"
        )
        documents = engine.search(case["query"])
        for rank, document in enumerate(documents[: args.limit], start=1):
            metadata = document.metadata or {}
            preview = " ".join(document.page_content.split())[:220]
            print(
                f"#{rank} source={metadata.get('source')} "
                f"row={metadata.get('row')} "
                f"rrf={metadata.get('rrf_score')} "
                f"reranker={metadata.get('reranker_score')}"
            )
            print(f"   {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
