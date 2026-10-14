"""Evaluate a fixed case set with forced retrieval-context injection.

Unlike the production /v1/chat path, this benchmark bypasses agent routing:
query -> B-node retrieval -> explicit context prompt -> Ollama answer -> judge.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from evaluation.evaluate_new_data_answers import (
    append_checkpoint,
    load_answer_cases,
    load_checkpoint,
    summarize,
)
from evaluation.evaluate_new_data_full import first_matching_rank
from evaluation.run_answer_ab import generate
from evaluation.score_answer_ab import deterministic_scores, judge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = Path(__file__).with_name("new_data_answer_cases_100_v2.json")
DEFAULT_OUTPUT = Path(__file__).with_name("results") / "forced_rag_answer_accuracy_100_v2.json"


def retrieve_direct(
    api_url: str,
    *,
    api_key: str,
    query: str,
    timeout: float,
) -> tuple[list[dict[str, Any]], float]:
    payload = json.dumps({"query": query}, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if api_key:
        headers["X-API-Key"] = api_key
    request = urllib.request.Request(api_url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:500]}") from exc
    if body.get("status") != "success":
        raise RuntimeError(f"检索 API 返回非 success: {body}")
    data = body.get("data", {})
    documents = data.get("documents")
    if not isinstance(documents, list):
        raise RuntimeError("检索 API 未返回 data.documents")
    return documents, float(data.get("elapsed_seconds") or 0.0)


def _with_retries(function, retries: int):
    error: str | None = None
    for attempt in range(1, retries + 2):
        try:
            return function(), None, attempt
        except (OSError, TimeoutError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            error = str(exc)
            if attempt <= retries:
                time.sleep(min(2 ** (attempt - 1), 5))
    return None, error, retries + 1


def evaluate_case(
    case: dict[str, Any],
    *,
    retrieve_api: str,
    retrieve_api_key: str,
    ollama_api: str,
    model: str,
    judge_model: str,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    retrieval, retrieval_error, retrieval_attempts = _with_retries(
        lambda: retrieve_direct(
            retrieve_api,
            api_key=retrieve_api_key,
            query=case["query"],
            timeout=timeout,
        ),
        retries,
    )
    base: dict[str, Any] = {
        "name": case["name"],
        "query": case["query"],
        "reference": case["reference"],
        "expected_source": case["expected_source"],
        "expected_row": case.get("expected_row"),
        "retrieval_attempts": retrieval_attempts,
        "retrieval_error": retrieval_error,
    }
    if retrieval is None:
        base["chat_error"] = retrieval_error
        base["generation_seconds"] = round(time.perf_counter() - started, 4)
        return base

    documents, retrieval_seconds = retrieval
    contexts = [
        str(document.get("content", "")).strip()
        for document in documents
        if str(document.get("content", "")).strip()
    ]
    context = "\n\n".join(
        f"[Context {index}]\n{text}" for index, text in enumerate(contexts, start=1)
    )
    flattened = []
    for rank, document in enumerate(documents, start=1):
        metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
        flattened.append(
            {
                "rank": rank,
                "source": metadata.get("source"),
                "row": metadata.get("row"),
                "model": metadata.get("model"),
                "reranker_score": metadata.get("reranker_score"),
            }
        )
    acceptable = case.get("acceptable_documents") or [
        {"source": case["expected_source"], "row": case.get("expected_row")}
    ]
    base.update(
        {
            "retrieval_seconds": retrieval_seconds,
            "retrieved": flattened,
            "retrieval_rank": first_matching_rank(flattened, acceptable),
            "context_count": len(contexts),
            "sources": [
                f"{item.get('source')}:{item.get('row')}" for item in flattened
            ],
            "retrieval_backend": "forced_remote",
        }
    )

    generated, generation_error, generation_attempts = _with_retries(
        lambda: generate(ollama_api, model, case["query"], context), retries
    )
    base["generation_attempts"] = generation_attempts
    if generated is None:
        base["chat_error"] = generation_error
        base["generation_seconds"] = round(time.perf_counter() - started, 4)
        return base

    answer = generated["answer"]
    base.update(
        {
            "answer": answer,
            "generation_seconds": round(float(generated["wall_seconds"]), 4),
            "tokens_per_second": generated.get("tokens_per_second"),
            "deterministic": deterministic_scores(case["reference"], answer),
        }
    )

    judge_started = time.perf_counter()
    judged, judge_error, judge_attempts = _with_retries(
        lambda: judge(
            ollama_api,
            judge_model,
            case["query"],
            case["reference"],
            answer,
        ),
        retries,
    )
    base.update(
        {
            "judge_seconds": round(time.perf_counter() - judge_started, 4),
            "judge_attempts": judge_attempts,
            "judge_error": judge_error,
        }
    )
    if judged is None:
        return base

    base["judge"] = judged
    base["semantic_pass"] = (
        judged["correctness"] >= 4
        and judged["completeness"] >= 4
        and judged["relevance"] >= 4
        and not judged["hallucination"]
    )
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description="Forced-RAG answer evaluation.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--retrieve-api", default=None)
    parser.add_argument("--retrieve-api-key", default=None)
    parser.add_argument("--ollama-api", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--model")
    parser.add_argument("--judge-model")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    retrieve_api = args.retrieve_api or (
        os.getenv("RETRIEVAL_API_URL", "http://127.0.0.1:8100").rstrip("/")
        + "/v1/retrieve"
    )
    retrieve_api_key = (
        args.retrieve_api_key
        if args.retrieve_api_key is not None
        else os.getenv("RETRIEVAL_API_KEY", "")
    )
    model = args.model or os.getenv("OLLAMA_MODEL", "enbot-v9:7b")
    judge_model = args.judge_model or os.getenv("AB_JUDGE_MODEL") or model
    cases, case_manifest = load_answer_cases(args.cases)
    output = args.output.resolve()
    checkpoint = (args.checkpoint or output.with_suffix(".jsonl")).resolve()

    completed = load_checkpoint(checkpoint) if args.resume else {}
    if not args.resume and checkpoint.exists():
        checkpoint.unlink()
    pending = [case for case in cases if case["name"] not in completed]

    print(f"评测集: {args.cases.resolve()}", flush=True)
    print(f"强制检索 API: {retrieve_api}", flush=True)
    print(f"生成模型/裁判模型: {model}/{judge_model}", flush=True)
    print(f"用例: {len(cases)}；已完成: {len(completed)}；待执行: {len(pending)}", flush=True)
    print(f"checkpoint: {checkpoint}", flush=True)

    for case in pending:
        result = evaluate_case(
            case,
            retrieve_api=retrieve_api,
            retrieve_api_key=retrieve_api_key,
            ollama_api=args.ollama_api,
            model=model,
            judge_model=judge_model,
            timeout=args.timeout,
            retries=args.retries,
        )
        completed[result["name"]] = result
        append_checkpoint(checkpoint, result)
        done = sum(item["name"] in completed for item in cases)
        if result.get("chat_error"):
            state = "ERROR"
            detail = result["chat_error"]
        elif result.get("judge_error"):
            state = "JUDGE_ERROR"
            detail = result["judge_error"]
        else:
            state = "PASS" if result["semantic_pass"] else "FAIL"
            scored = result["judge"]
            detail = (
                f"rank={result['retrieval_rank'] or '-'} "
                f"correctness={scored['correctness']}/5 "
                f"complete={scored['completeness']}/5 "
                f"relevant={scored['relevance']}/5 "
                f"hallucination={scored['hallucination']}"
            )
        print(f"[{done}/{len(cases)}] {state} {case['name']} {detail}", flush=True)

    ordered = [completed[case["name"]] for case in cases if case["name"] in completed]
    summary = summarize(cases, ordered)
    retrieval_success = [item for item in ordered if not item.get("retrieval_error")]
    summary["retrieval_recall_at_1"] = (
        sum(item.get("retrieval_rank") == 1 for item in retrieval_success) / len(retrieval_success)
        if retrieval_success
        else None
    )
    summary["retrieval_recall_at_2"] = (
        sum(0 < int(item.get("retrieval_rank") or 0) <= 2 for item in retrieval_success)
        / len(retrieval_success)
        if retrieval_success
        else None
    )
    report = {
        "schema_version": 1,
        "mode": "forced_remote_retrieval_context_injection",
        "cases": case_manifest,
        "retrieve_api": retrieve_api,
        "model": model,
        "judge_model": judge_model,
        "pass_rule": "correctness>=4, completeness>=4, relevance>=4, hallucination=false",
        "summary": summary,
        "failures": [
            item
            for item in ordered
            if item.get("chat_error")
            or item.get("judge_error")
            or item.get("semantic_pass") is False
        ],
        "results": ordered,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    accuracy = summary.get("semantic_accuracy")
    print("", flush=True)
    print(
        "Summary: "
        f"judged={summary['judged_answers']}/{summary['requested_cases']}, "
        f"passed={summary['semantic_passed']}, accuracy={accuracy:.4f}, "
        f"retrieval_R@1={summary['retrieval_recall_at_1']:.4f}, "
        f"retrieval_R@2={summary['retrieval_recall_at_2']:.4f}, "
        f"hallucination_rate={summary['hallucination_rate']:.4f}, "
        f"errors={summary['chat_errors'] + summary['judge_errors']}",
        flush=True,
    )
    print(f"报告: {output}", flush=True)
    return 2 if summary["chat_errors"] or summary["judge_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())