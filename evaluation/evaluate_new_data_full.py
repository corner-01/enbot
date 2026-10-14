"""Evaluate retrieval for every labeled row in the current knowledge base.

This benchmark exercises only the running retrieval API. It does not call the
chat model, so the scores describe retrieval coverage rather than answer
quality. Progress is written to stdout and every completed case is checkpointed
as JSONL so an interrupted full run can be resumed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from dotenv import load_dotenv

from evaluation.build_retrieval_cases_v4 import (
    TXT_CASES,
    field_text,
    normalize_query,
    select_qa_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(__file__).with_name("results") / "new_data_full_retrieval.json"
_PRINT_LOCK = threading.Lock()


def normalize_source(value: Any) -> str:
    """Normalize Windows/Linux separators without changing source identity."""
    return str(value or "").replace("\\", "/").strip()


def _document_key(document: dict[str, Any]) -> tuple[str, int | None]:
    row = document.get("row")
    try:
        row = int(row) if row is not None else None
    except (TypeError, ValueError):
        row = None
    return normalize_source(document.get("source")), row


def _accepted_key(document: dict[str, Any]) -> tuple[str, int | None]:
    row = document.get("row")
    return normalize_source(document.get("source")), int(row) if row is not None else None


def first_matching_rank(
    documents: list[dict[str, Any]], acceptable_documents: Iterable[dict[str, Any]]
) -> int:
    accepted = {_accepted_key(item) for item in acceptable_documents}
    for rank, document in enumerate(documents, start=1):
        if _document_key(document) in accepted:
            return rank
    return 0


def _qa_key(question: str, answer: str) -> str:
    return f"{normalize_query(question)}\u241f{normalize_query(answer)}"


def build_cases(knowledge_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one positive case for every valid XLSX row plus the TXT probes."""
    root = knowledge_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"知识库目录不存在: {root}")

    cases: list[dict[str, Any]] = []
    equivalent_locations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    xlsx_paths = [
        path for path in sorted(root.rglob("*.xlsx")) if not path.name.startswith("~$")
    ]
    for path in xlsx_paths:
        source = str(path.relative_to(root))
        try:
            dataframe = pd.read_excel(path, engine="openpyxl")
            question_column, answer_column = select_qa_columns(dataframe)
        except Exception as exc:
            skipped["unreadable_xlsx"] += 1
            print(f"[WARN] 无法读取 {source}: {exc}", flush=True)
            continue

        for row_index, row in dataframe.iterrows():
            question = field_text(row.get(question_column))
            answer = field_text(row.get(answer_column))
            if not question:
                skipped["empty_question"] += 1
                continue
            if not answer:
                skipped["empty_answer"] += 1
                continue
            location = {"source": source, "row": int(row_index) + 2}
            key = _qa_key(question, answer)
            equivalent_locations[key].append(location)
            cases.append(
                {
                    "query": question,
                    "reference": answer,
                    "expected_source": source,
                    "expected_row": int(row_index) + 2,
                    "source_kind": "xlsx",
                    "_qa_key": key,
                }
            )
            source_counts[source] += 1

    available_txt = {str(path.relative_to(root)) for path in root.rglob("*.txt")}
    for source, query in TXT_CASES.items():
        if source not in available_txt:
            skipped["missing_txt_probe"] += 1
            print(f"[WARN] 未找到 TXT 评测来源: {source}", flush=True)
            continue
        cases.append(
            {
                "query": query,
                "expected_source": source,
                "source_kind": "txt",
                "acceptable_documents": [{"source": source, "row": None}],
            }
        )
        source_counts[source] += 1

    for index, case in enumerate(cases, start=1):
        case["name"] = f"NEW-{index:05d}"
        key = case.pop("_qa_key", None)
        if key:
            case["acceptable_documents"] = equivalent_locations[key]

    manifest = {
        "knowledge_root": str(root),
        "xlsx_file_count": len(xlsx_paths),
        "case_count": len(cases),
        "source_count": len(source_counts),
        "source_case_counts": dict(sorted(source_counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "label_rule": "XLSX row uses pandas index + 2; exact duplicate question+answer locations are equivalent",
    }
    return cases, manifest


def retrieve_from_api(
    api_url: str,
    query: str,
    *,
    api_key: str,
    timeout: float,
) -> list[dict[str, Any]]:
    payload = json.dumps(
        {"query": query, "include_content": False}, ensure_ascii=False
    ).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if api_key:
        headers["X-API-Key"] = api_key
    request = urllib.request.Request(api_url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    if body.get("status") != "success":
        raise RuntimeError(f"API 返回非 success: {body}")
    documents = body.get("data", {}).get("retrieval_debug")
    if not isinstance(documents, list):
        raise RuntimeError("API 未返回 data.retrieval_debug，请确认 /v1/retrieve 服务已重启")
    return documents


def evaluate_case(
    case: dict[str, Any],
    *,
    api_url: str,
    api_key: str,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    error = ""
    documents: list[dict[str, Any]] = []
    attempts = 0
    for attempts in range(1, retries + 2):
        try:
            documents = retrieve_from_api(
                api_url,
                case["query"],
                api_key=api_key,
                timeout=timeout,
            )
            error = ""
            break
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
            error = str(exc)
            if attempts <= retries:
                time.sleep(min(2 ** (attempts - 1), 5))

    strict_expected = [
        {"source": case["expected_source"], "row": case.get("expected_row")}
    ]
    strict_rank = first_matching_rank(documents, strict_expected) if not error else 0
    equivalent_rank = (
        first_matching_rank(documents, case["acceptable_documents"]) if not error else 0
    )
    compact_documents = [
        {
            "rank": document.get("rank", rank),
            "source": document.get("source"),
            "row": document.get("row"),
            "model": document.get("model"),
            "reranker_score": document.get("reranker_score"),
            "preview": str(document.get("preview", ""))[:300],
        }
        for rank, document in enumerate(documents, start=1)
    ]
    return {
        "name": case["name"],
        "query": case["query"],
        "expected_source": case["expected_source"],
        "expected_row": case.get("expected_row"),
        "strict_rank": strict_rank,
        "equivalent_rank": equivalent_rank,
        "returned_count": len(documents),
        "duration_seconds": round(time.perf_counter() - started, 4),
        "attempts": attempts,
        "error": error or None,
        "documents": compact_documents,
    }


def _metric_block(ranks: list[int]) -> dict[str, Any]:
    total = len(ranks)
    return {
        "recall_at_1": sum(rank == 1 for rank in ranks) / total if total else 0.0,
        "recall_at_2": sum(0 < rank <= 2 for rank in ranks) / total if total else 0.0,
        "mrr": sum((1.0 / rank) if rank else 0.0 for rank in ranks) / total if total else 0.0,
        "found": sum(rank > 0 for rank in ranks),
        "not_found": sum(rank == 0 for rank in ranks),
        "total": total,
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(
    cases: list[dict[str, Any]], results: list[dict[str, Any]]
) -> dict[str, Any]:
    case_by_name = {case["name"]: case for case in cases}
    successful = [result for result in results if not result.get("error")]
    errors = [result for result in results if result.get("error")]
    durations = [float(result["duration_seconds"]) for result in successful]
    per_source_ranks: dict[str, list[int]] = defaultdict(list)
    for result in successful:
        source = case_by_name[result["name"]]["expected_source"]
        per_source_ranks[source].append(int(result["equivalent_rank"]))

    return {
        "requested_cases": len(cases),
        "completed_cases": len(results),
        "successful_api_calls": len(successful),
        "api_errors": len(errors),
        "strict": _metric_block([int(item["strict_rank"]) for item in successful]),
        "equivalent": _metric_block(
            [int(item["equivalent_rank"]) for item in successful]
        ),
        "latency_seconds": {
            "mean": statistics.fmean(durations) if durations else 0.0,
            "p50": _percentile(durations, 0.50),
            "p95": _percentile(durations, 0.95),
            "max": max(durations, default=0.0),
        },
        "per_source": {
            source: _metric_block(ranks)
            for source, ranks in sorted(per_source_ranks.items())
        },
    }


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            completed[item["name"]] = item
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"[WARN] 忽略 checkpoint 第 {line_number} 行: {exc}", flush=True)
    return completed


def _append_checkpoint(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate every valid QA row in the current RAG knowledge source."
    )
    parser.add_argument("--api", default="http://127.0.0.1:8000/v1/retrieve")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--knowledge-root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cases-output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers 必须至少为 1")
    if args.limit < 0:
        parser.error("--limit 不能小于 0")

    load_dotenv(PROJECT_ROOT / ".env")
    knowledge_root = args.knowledge_root or Path(
        os.getenv("KNOWLEDGE_FILE", PROJECT_ROOT / "knowledge_data")
    )
    api_key = args.api_key if args.api_key is not None else os.getenv("ENBOT_API_KEY", "")
    output = args.output.resolve()
    checkpoint = (args.checkpoint or output.with_suffix(".jsonl")).resolve()
    cases_output = (
        args.cases_output
        or output.with_name(f"{output.stem}_cases.json")
    ).resolve()

    all_cases, manifest = build_cases(knowledge_root)
    cases = all_cases[: args.limit] if args.limit else all_cases
    output.parent.mkdir(parents=True, exist_ok=True)
    cases_output.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    prior = _load_checkpoint(checkpoint) if args.resume else {}
    if not args.resume and checkpoint.exists():
        checkpoint.unlink()
    pending = [case for case in cases if case["name"] not in prior]

    print(f"知识库: {Path(knowledge_root).expanduser().resolve()}", flush=True)
    print(f"检索 API: {args.api}", flush=True)
    print(f"评测用例: {len(cases)}（完整可用 {len(all_cases)}）", flush=True)
    print(f"并发数: {args.workers}；已完成: {len(prior)}；待执行: {len(pending)}", flush=True)
    print(f"checkpoint: {checkpoint}", flush=True)

    completed = dict(prior)
    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map: dict[Future[dict[str, Any]], dict[str, Any]] = {
                executor.submit(
                    evaluate_case,
                    case,
                    api_url=args.api,
                    api_key=api_key,
                    timeout=args.timeout,
                    retries=args.retries,
                ): case
                for case in pending
            }
            done_count = len(prior)
            for future in as_completed(future_map):
                result = future.result()
                completed[result["name"]] = result
                _append_checkpoint(checkpoint, result)
                done_count += 1
                if result.get("error"):
                    state = "ERROR"
                    detail = result["error"]
                else:
                    state = "PASS" if result["equivalent_rank"] else "MISS"
                    detail = (
                        f"rank={result['equivalent_rank'] or '-'} "
                        f"strict={result['strict_rank'] or '-'} "
                        f"{result['duration_seconds']:.2f}s"
                    )
                with _PRINT_LOCK:
                    print(
                        f"[{done_count}/{len(cases)}] {state} {result['name']} "
                        f"{detail} | {result['query'][:90]}",
                        flush=True,
                    )

    ordered_results = [completed[case["name"]] for case in cases if case["name"] in completed]
    summary = summarize(cases, ordered_results)
    report = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "api": args.api,
        "manifest": manifest,
        "summary": summary,
        "failures": [
            item
            for item in ordered_results
            if item.get("error") or not item.get("equivalent_rank")
        ],
        "results": ordered_results,
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    strict = summary["strict"]
    equivalent = summary["equivalent"]
    latency = summary["latency_seconds"]
    print("", flush=True)
    print(
        "Summary: "
        f"completed={summary['completed_cases']}/{summary['requested_cases']}, "
        f"api_errors={summary['api_errors']}, "
        f"strict Recall@1={strict['recall_at_1']:.4f}, "
        f"strict Recall@2={strict['recall_at_2']:.4f}, "
        f"equivalent Recall@1={equivalent['recall_at_1']:.4f}, "
        f"equivalent Recall@2={equivalent['recall_at_2']:.4f}, "
        f"MRR={equivalent['mrr']:.4f}, "
        f"latency p50/p95={latency['p50']:.3f}/{latency['p95']:.3f}s",
        flush=True,
    )
    print(f"报告: {output}", flush=True)
    print(f"用例: {cases_output}", flush=True)
    return 2 if summary["api_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())