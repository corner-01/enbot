"""Tune EnBot hybrid retrieval on a development set with Optuna.

This program refuses holdout paths. It loads the embedding model, Milvus
collection and reranker once, then reconfigures only retrieval hyperparameters
between trials.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna
from dotenv import load_dotenv
from optuna.samplers import TPESampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.knowledge_retriever import get_knowledge_retriever


DEFAULT_CASES = Path(__file__).with_name("retrieval_cases_v3.json")
DEFAULT_OUTPUT = Path(__file__).parent / "optuna" / "retrieval_v1"
STUDY_NAME = "enbot-retrieval-v1"


def load_cases(path: Path) -> list[dict[str, Any]]:
    resolved = path.resolve()
    if "holdout" in {part.lower() for part in resolved.parts}:
        raise ValueError("Optuna禁止使用holdout验证集调参")
    cases = json.loads(resolved.read_text(encoding="utf-8-sig"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("开发集必须是非空JSON数组")
    return cases


def relevant_rank(case: dict[str, Any], documents: list[Any]) -> int:
    expected_source = str(case.get("expected_source", ""))
    expected_row = case.get("expected_row")
    acceptable_documents = case.get("acceptable_documents") or [
        {"source": expected_source, "row": expected_row}
    ]
    for rank, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        document_matches = any(
            (not expected.get("source") or metadata.get("source") == expected["source"])
            and (expected.get("row") is None or metadata.get("row") == expected["row"])
            for expected in acceptable_documents
        )
        if document_matches:
            return rank
    return 0


def evaluate(engine: Any, cases: list[dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    passed = 0
    positives = 0
    negatives = 0
    negative_correct = 0
    recall_at_1_count = 0
    recall_at_2_count = 0
    reciprocal_rank_sum = 0.0
    failed_cases: list[dict[str, Any]] = []

    for case in cases:
        documents = engine.search(case["query"])
        if case.get("expect_empty"):
            negatives += 1
            correct = not documents
            negative_correct += int(correct)
            passed += int(correct)
            if not correct:
                failed_cases.append(
                    {
                        "name": case["name"],
                        "failure": "expected_empty_but_returned_documents",
                        "query": case["query"],
                        "returned": [
                            {
                                "source": document.metadata.get("source"),
                                "row": document.metadata.get("row"),
                                "reranker_score": document.metadata.get(
                                    "reranker_score"
                                ),
                            }
                            for document in documents
                        ],
                    }
                )
            continue

        positives += 1
        rank = relevant_rank(case, documents)
        passed += int(rank > 0)
        recall_at_1_count += int(rank == 1)
        recall_at_2_count += int(0 < rank <= 2)
        reciprocal_rank_sum += 1.0 / rank if rank else 0.0
        if rank == 0:
            failed_cases.append(
                {
                    "name": case["name"],
                    "failure": "expected_document_not_returned",
                    "query": case["query"],
                    "expected_source": case.get("expected_source"),
                    "expected_row": case.get("expected_row"),
                    "returned": [
                        {
                            "source": document.metadata.get("source"),
                            "row": document.metadata.get("row"),
                            "reranker_score": document.metadata.get(
                                "reranker_score"
                            ),
                        }
                        for document in documents
                    ],
                }
            )

    elapsed = time.perf_counter() - started
    return {
        "pass_rate": passed / len(cases),
        "recall_at_1": recall_at_1_count / positives if positives else 0.0,
        "recall_at_2": recall_at_2_count / positives if positives else 0.0,
        "mrr": reciprocal_rank_sum / positives if positives else 0.0,
        "no_answer_accuracy": (
            negative_correct / negatives if negatives else 0.0
        ),
        "elapsed_seconds": elapsed,
        "milliseconds_per_case": elapsed * 1000.0 / len(cases),
        "failed_cases": failed_cases,
    }


def configure(engine: Any, vectorstore: Any, params: dict[str, Any]) -> None:
    dense_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": int(params["dense_k"])},
    )
    engine.configure(
        dense_retriever=dense_retriever,
        dense_k=int(params["dense_k"]),
        lexical_k=int(params["lexical_k"]),
        rerank_k=int(params["rerank_k"]),
        final_k=2,
        rrf_k=float(params["rrf_k"]),
        dense_weight=float(params["dense_weight"]),
        lexical_weight=float(params["lexical_weight"]),
        rerank_relative_threshold=float(params["rerank_relative_threshold"]),
        rerank_absolute_threshold=float(params["rerank_absolute_threshold"]),
        rerank_min_score=float(params["rerank_min_score"]),
    )


def current_params(engine: Any) -> dict[str, Any]:
    return {
        "dense_k": int(engine.dense_k),
        "lexical_k": int(engine.lexical_k),
        "rerank_k": int(engine.rerank_k),
        "rrf_k": float(engine.rrf_k),
        "dense_weight": float(engine.dense_weight),
        "lexical_weight": float(engine.lexical_weight),
        "rerank_relative_threshold": float(
            engine.rerank_relative_threshold
        ),
        "rerank_absolute_threshold": float(
            engine.rerank_absolute_threshold
        ),
        "rerank_min_score": float(engine.rerank_min_score),
    }


def suggested_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "dense_k": trial.suggest_categorical("dense_k", [20, 30, 40, 50]),
        "lexical_k": trial.suggest_categorical(
            "lexical_k", [20, 30, 40, 50, 60]
        ),
        "rerank_k": trial.suggest_categorical(
            "rerank_k", [5, 8, 10, 15, 20]
        ),
        "rrf_k": trial.suggest_categorical("rrf_k", [20.0, 40.0, 60.0, 80.0]),
        "dense_weight": trial.suggest_categorical(
            "dense_weight", [0.5, 0.75, 1.0, 1.25, 1.5]
        ),
        "lexical_weight": trial.suggest_categorical(
            "lexical_weight", [0.5, 0.75, 1.0, 1.25, 1.5]
        ),
        "rerank_relative_threshold": trial.suggest_categorical(
            "rerank_relative_threshold", [0.6, 0.7, 0.8, 0.9]
        ),
        "rerank_absolute_threshold": trial.suggest_categorical(
            "rerank_absolute_threshold", [0.0, 0.05, 0.1, 0.15]
        ),
        "rerank_min_score": trial.suggest_categorical(
            "rerank_min_score", [-1.0, 0.0, 0.02, 0.05, 0.08, 0.1]
        ),
    }


def score_metrics(
    metrics: dict[str, Any], baseline: dict[str, Any]
) -> float:
    score = (
        0.35 * metrics["recall_at_1"]
        + 0.15 * metrics["recall_at_2"]
        + 0.15 * metrics["mrr"]
        + 0.25 * metrics["no_answer_accuracy"]
        + 0.10 * metrics["pass_rate"]
    )

    # A trial may improve ranking only by rejecting valid questions. Apply
    # strong penalties when safety or recall falls below the frozen baseline.
    if metrics["no_answer_accuracy"] < baseline["no_answer_accuracy"]:
        score -= 5.0 * (
            baseline["no_answer_accuracy"] - metrics["no_answer_accuracy"]
        )
    recall_floor = baseline["recall_at_2"] - 0.02
    if metrics["recall_at_2"] < recall_floor:
        score -= 5.0 * (recall_floor - metrics["recall_at_2"])

    baseline_latency = max(baseline["milliseconds_per_case"], 1.0)
    latency_ratio = metrics["milliseconds_per_case"] / baseline_latency
    score -= 0.02 * max(0.0, latency_ratio - 1.0)
    return score


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    cases = load_cases(args.cases)
    args.output.mkdir(parents=True, exist_ok=True)

    retriever = get_knowledge_retriever(raise_on_error=True)
    if retriever is None or retriever.hybrid_engine is None:
        raise RuntimeError("混合检索器初始化失败")
    engine = retriever.hybrid_engine
    vectorstore = retriever.vectorstore

    baseline_params = current_params(engine)
    configure(engine, vectorstore, baseline_params)
    baseline_metrics = evaluate(engine, cases)
    baseline = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases": str(args.cases.resolve()),
        "params": baseline_params,
        "metrics": baseline_metrics,
    }
    json_dump(args.output / "baseline.json", baseline)
    print("Baseline:")
    print(json.dumps(baseline, ensure_ascii=False, indent=2))

    if args.baseline_only:
        print("Baseline-only run completed; Optuna was not started.")
        return 0
    if args.trials < 1:
        raise ValueError("trials必须大于0")

    database_path = (args.output / "study.db").resolve()
    storage = f"sqlite:///{database_path.as_posix()}"
    sampler = TPESampler(seed=20260928, n_startup_trials=10, multivariate=True)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    if not study.trials:
        study.enqueue_trial(baseline_params)

    def objective(trial: optuna.Trial) -> float:
        params = suggested_params(trial)
        configure(engine, vectorstore, params)
        metrics = evaluate(engine, cases)
        score = score_metrics(metrics, baseline_metrics)
        for name, value in metrics.items():
            trial.set_user_attr(name, value)
        trial.set_user_attr("score", score)
        print(
            f"trial={trial.number} score={score:.6f} "
            f"R@1={metrics['recall_at_1']:.4f} "
            f"R@2={metrics['recall_at_2']:.4f} "
            f"reject={metrics['no_answer_accuracy']:.4f} "
            f"ms/case={metrics['milliseconds_per_case']:.1f}"
        )
        return score

    study.optimize(objective, n_trials=args.trials, n_jobs=1, gc_after_trial=True)
    best = study.best_trial
    best_metrics = {
        key: value
        for key, value in best.user_attrs.items()
        if key != "score"
    }
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "study": STUDY_NAME,
        "cases": str(args.cases.resolve()),
        "trials_total": len(study.trials),
        "best_trial": best.number,
        "best_score": best.value,
        "best_params": best.params,
        "best_metrics": best_metrics,
        "baseline_params": baseline_params,
        "baseline_metrics": baseline_metrics,
        "holdout_used": False,
    }
    json_dump(args.output / "best_result.json", result)
    json_dump(args.output / "best_params.json", best.params)
    print("Best result:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
