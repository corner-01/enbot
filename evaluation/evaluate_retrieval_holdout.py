"""Evaluate a locked retrieval holdout exactly once through the local API."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.retrieval_benchmark import (
    first_relevant_rank,
    load_cases,
    retrieve_from_api,
)


DEFAULT_CASES = Path(__file__).parent / "holdout" / "retrieval_holdout_v2.json"
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "retrieval_holdout_v2_result.json"
ACCEPTANCE_THRESHOLDS = {
    "pass_rate": 0.90,
    "recall_at_1": 0.85,
    "recall_at_2": 0.90,
    "mrr": 0.875,
    "no_answer_accuracy": 0.90,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_document(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": document.get("rank"),
        "source": document.get("source"),
        "row": document.get("row"),
        "model": document.get("model"),
        "reranker_score": document.get("reranker_score"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest_path = args.cases.with_suffix(".manifest.json")
    if not args.cases.exists() or not manifest_path.exists():
        raise FileNotFoundError("Build and lock Holdout V2 before evaluating it.")
    if args.output.exists():
        raise FileExistsError(f"Holdout result already exists: {args.output}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "LOCKED_UNEVALUATED":
        raise RuntimeError(
            f"Holdout cannot be evaluated again; status={manifest.get('status')}"
        )
    actual_digest = sha256_file(args.cases)
    if actual_digest != manifest.get("sha256"):
        raise RuntimeError("Holdout SHA256 mismatch; the locked file was modified.")

    cases = load_cases(args.cases)
    passed = 0
    positive_count = 0
    negative_count = 0
    negative_correct = 0
    recall_at_1_count = 0
    recall_at_2_count = 0
    reciprocal_rank_sum = 0.0
    details: list[dict[str, Any]] = []

    for case in cases:
        documents = retrieve_from_api(args.api, case["query"])
        if case.get("expect_empty"):
            negative_count += 1
            case_passed = not documents
            negative_correct += int(case_passed)
            rank = 0
        else:
            positive_count += 1
            rank = first_relevant_rank(case, documents)
            case_passed = rank > 0
            recall_at_1_count += int(rank == 1)
            recall_at_2_count += int(0 < rank <= 2)
            reciprocal_rank_sum += 1.0 / rank if rank else 0.0
        passed += int(case_passed)
        details.append(
            {
                "name": case["name"],
                "passed": case_passed,
                "relevant_rank": rank,
                "returned": [compact_document(item) for item in documents],
            }
        )
        print(
            f"[{'PASS' if case_passed else 'FAIL'}] {case['name']}: "
            f"rank={rank or '-'}, returned={len(documents)}"
        )

    total = len(cases)
    metrics = {
        "passed": passed,
        "total": total,
        "pass_rate": passed / total if total else 0.0,
        "recall_at_1": recall_at_1_count / positive_count if positive_count else 0.0,
        "recall_at_2": recall_at_2_count / positive_count if positive_count else 0.0,
        "mrr": reciprocal_rank_sum / positive_count if positive_count else 0.0,
        "no_answer_accuracy": (
            negative_correct / negative_count if negative_count else 0.0
        ),
        "positive_cases": positive_count,
        "negative_cases": negative_count,
        "failed_cases": total - passed,
    }
    checks = {
        metric: metrics[metric] >= threshold
        for metric, threshold in ACCEPTANCE_THRESHOLDS.items()
    }
    gate_passed = all(checks.values())
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if gate_passed else "FAIL",
        "candidate": "optuna-retrieval-candidate-v1",
        "cases": str(args.cases.resolve()),
        "cases_sha256": actual_digest,
        "api": args.api,
        "acceptance_thresholds": ACCEPTANCE_THRESHOLDS,
        "acceptance_checks": checks,
        "metrics": metrics,
        "case_results": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest["status"] = "EVALUATED"
    manifest["evaluated_at"] = result["created_at"]
    manifest["result"] = str(args.output.resolve())
    manifest["result_sha256"] = sha256_file(args.output)
    manifest["acceptance_thresholds"] = ACCEPTANCE_THRESHOLDS
    manifest["acceptance_checks"] = checks
    manifest["release_gate"] = result["status"]
    manifest["metrics"] = metrics
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nHoldout V2 summary")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Release gate: {result['status']}")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    print(f"Result: {args.output}")
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
