"""Build a deterministic, source-balanced 100-case answer evaluation set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from evaluation.build_retrieval_cases_v4 import normalize_query
from evaluation.evaluate_new_data_full import build_cases


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(__file__).with_name("new_data_answer_cases_100.json")


def _identity(case: dict[str, Any]) -> str:
    return (
        normalize_query(str(case.get("query", "")))
        + "\u241f"
        + normalize_query(str(case.get("reference", "")))
    )


def _stable_order(case: dict[str, Any], seed: int) -> str:
    value = (
        f"{seed}|{case.get('expected_source', '')}|{case.get('expected_row', '')}|"
        f"{case.get('query', '')}"
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def balanced_sample(
    cases: list[dict[str, Any]], count: int, *, seed: int = 42
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Deduplicate Q&A pairs and sample round-robin across source files."""
    if count < 1:
        raise ValueError("count must be positive")

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_count = 0
    for case in cases:
        if not case.get("reference"):
            continue
        identity = _identity(case)
        if not identity or identity in seen:
            duplicate_count += 1
            continue
        seen.add(identity)
        unique.append(case)

    if count > len(unique):
        raise ValueError(f"只找到 {len(unique)} 条唯一问答，无法抽取 {count} 条")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in unique:
        grouped[str(case["expected_source"])].append(case)
    for source_cases in grouped.values():
        source_cases.sort(key=lambda item: _stable_order(item, seed))

    source_order = sorted(
        grouped,
        key=lambda source: hashlib.sha256(f"{seed}|{source}".encode("utf-8")).hexdigest(),
    )
    queues = {source: deque(grouped[source]) for source in source_order}
    active = deque(source_order)
    selected: list[dict[str, Any]] = []
    while active and len(selected) < count:
        source = active.popleft()
        queue = queues[source]
        selected.append(dict(queue.popleft()))
        if queue:
            active.append(source)

    for index, case in enumerate(selected, start=1):
        case["original_name"] = case.get("name")
        case["name"] = f"RAG100-{index:03d}"

    stats = {
        "input_cases": len(cases),
        "unique_answer_cases": len(unique),
        "duplicates_removed": duplicate_count,
        "selected_cases": len(selected),
        "covered_sources": len({case["expected_source"] for case in selected}),
    }
    return selected, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a balanced RAG answer sample.")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--knowledge-root", type=Path)
    parser.add_argument("--exclude-cases", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    knowledge_root = args.knowledge_root or Path(
        os.getenv("KNOWLEDGE_FILE", PROJECT_ROOT / "knowledge_data")
    )
    all_cases, source_manifest = build_cases(Path(knowledge_root))
    excluded_count = 0
    if args.exclude_cases:
        excluded_payload = json.loads(args.exclude_cases.read_text(encoding="utf-8-sig"))
        excluded_cases = (
            excluded_payload.get("cases", [])
            if isinstance(excluded_payload, dict)
            else excluded_payload
        )
        excluded_identities = {
            _identity(case) for case in excluded_cases if isinstance(case, dict)
        }
        before_exclusion = len(all_cases)
        all_cases = [case for case in all_cases if _identity(case) not in excluded_identities]
        excluded_count = before_exclusion - len(all_cases)
    selected, stats = balanced_sample(all_cases, args.count, seed=args.seed)
    stats["excluded_previous_cases"] = excluded_count
    source_counts = Counter(case["expected_source"] for case in selected)
    payload = {
        "schema_version": 1,
        "purpose": "Source-balanced answer accuracy evaluation from the current RAG corpus",
        "seed": args.seed,
        "selection": "global Q&A deduplication, deterministic per-source ordering, round-robin sources",
        "knowledge_root": str(Path(knowledge_root).expanduser().resolve()),
        "source_manifest": source_manifest,
        "stats": stats,
        "selected_source_counts": dict(sorted(source_counts.items())),
        "cases": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"输出: {args.output.resolve()}")
    print(
        f"抽取 {len(selected)} 条，覆盖 {stats['covered_sources']} 个来源，"
        f"去除重复 {stats['duplicates_removed']} 条"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())