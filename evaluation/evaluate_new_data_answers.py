"""Evaluate final EnBot answers against every reference answer in the new data.

The running /v1/chat endpoint produces each candidate answer. A deterministic
reference comparison and a blinded local Ollama judge then score correctness,
completeness, relevance, and hallucination. Results are checkpointed after each
case so a long run can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from evaluation.evaluate_new_data_full import build_cases
from evaluation.score_answer_ab import deterministic_scores, judge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(__file__).with_name("results") / "new_data_answer_accuracy.json"


def post_chat(
    api_url: str,
    *,
    api_key: str,
    query: str,
    session_id: str,
    timeout: float,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "query": query,
            "session_id": session_id,
            "evaluate": False,
            "debug": False,
        },
        ensure_ascii=False,
    ).encode("utf-8")
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
        raise RuntimeError(f"聊天 API 返回非 success: {body}")
    answer = str(body.get("data", {}).get("response", "")).strip()
    if not answer:
        raise RuntimeError("聊天 API 返回了空答案")
    return {
        "answer": answer,
        "request_id": body.get("request_id"),
        "sources": body.get("data", {}).get("sources", []),
        "retrieval_backend": body.get("data", {}).get("retrieval_backend"),
    }


def _retry_chat(
    case: dict[str, Any],
    *,
    api_url: str,
    api_key: str,
    timeout: float,
    retries: int,
) -> tuple[dict[str, Any] | None, str | None, int, float]:
    started = time.perf_counter()
    error: str | None = None
    for attempt in range(1, retries + 2):
        try:
            result = post_chat(
                api_url,
                api_key=api_key,
                query=case["query"],
                session_id=f"answer-eval-{case['name'].lower()}",
                timeout=timeout,
            )
            return result, None, attempt, time.perf_counter() - started
        except (OSError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            error = str(exc)
            if attempt <= retries:
                time.sleep(min(2 ** (attempt - 1), 5))
    return None, error, retries + 1, time.perf_counter() - started


def _retry_judge(
    *,
    ollama_url: str,
    model: str,
    query: str,
    reference: str,
    candidate: str,
    retries: int,
) -> tuple[dict[str, Any] | None, str | None, int, float]:
    started = time.perf_counter()
    error: str | None = None
    for attempt in range(1, retries + 2):
        try:
            result = judge(ollama_url, model, query, reference, candidate)
            return result, None, attempt, time.perf_counter() - started
        except (OSError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            error = str(exc)
            if attempt <= retries:
                time.sleep(min(2 ** (attempt - 1), 5))
    return None, error, retries + 1, time.perf_counter() - started


def evaluate_case(
    case: dict[str, Any],
    *,
    api_url: str,
    api_key: str,
    ollama_url: str,
    judge_model: str,
    timeout: float,
    retries: int,
    skip_judge: bool,
) -> dict[str, Any]:
    chat, chat_error, chat_attempts, generation_seconds = _retry_chat(
        case,
        api_url=api_url,
        api_key=api_key,
        timeout=timeout,
        retries=retries,
    )
    base: dict[str, Any] = {
        "name": case["name"],
        "query": case["query"],
        "reference": case["reference"],
        "expected_source": case["expected_source"],
        "expected_row": case.get("expected_row"),
        "generation_seconds": round(generation_seconds, 4),
        "chat_attempts": chat_attempts,
        "chat_error": chat_error,
    }
    if chat is None:
        return base

    answer = chat["answer"]
    base.update(
        {
            "answer": answer,
            "request_id": chat["request_id"],
            "sources": chat["sources"],
            "retrieval_backend": chat["retrieval_backend"],
            "deterministic": deterministic_scores(case["reference"], answer),
        }
    )
    if skip_judge:
        return base

    judged, judge_error, judge_attempts, judge_seconds = _retry_judge(
        ollama_url=ollama_url,
        model=judge_model,
        query=case["query"],
        reference=case["reference"],
        candidate=answer,
        retries=retries,
    )
    base.update(
        {
            "judge_seconds": round(judge_seconds, 4),
            "judge_attempts": judge_attempts,
            "judge_error": judge_error,
        }
    )
    if judged is None:
        return base

    semantic_pass = (
        judged["correctness"] >= 4
        and judged["completeness"] >= 4
        and judged["relevance"] >= 4
        and not judged["hallucination"]
    )
    base["judge"] = judged
    base["semantic_pass"] = semantic_pass
    return base


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(cases: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    generated = [item for item in results if item.get("answer")]
    judged = [item for item in generated if item.get("judge")]
    passed = [item for item in judged if item.get("semantic_pass")]
    deterministic = [item["deterministic"] for item in generated]
    per_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in judged:
        per_source[item["expected_source"]].append(item)

    def metric_average(name: str) -> float | None:
        values = [float(item["judge"][name]) for item in judged]
        return _mean(values)

    def deterministic_average(name: str) -> float | None:
        values = [
            float(item[name])
            for item in deterministic
            if item.get(name) is not None
        ]
        return _mean(values)

    generation_times = [float(item["generation_seconds"]) for item in generated]
    judge_times = [float(item["judge_seconds"]) for item in judged]
    requested = len(cases)
    return {
        "requested_cases": requested,
        "completed_cases": len(results),
        "generated_answers": len(generated),
        "judged_answers": len(judged),
        "chat_errors": sum(bool(item.get("chat_error")) for item in results),
        "judge_errors": sum(bool(item.get("judge_error")) for item in generated),
        "semantic_passed": len(passed),
        "semantic_accuracy": len(passed) / len(judged) if judged else None,
        "completion_adjusted_accuracy": len(passed) / requested if requested else None,
        "average_correctness_1_to_5": metric_average("correctness"),
        "average_completeness_1_to_5": metric_average("completeness"),
        "average_relevance_1_to_5": metric_average("relevance"),
        "hallucination_rate": (
            sum(bool(item["judge"]["hallucination"]) for item in judged) / len(judged)
            if judged
            else None
        ),
        "average_token_f1": deterministic_average("token_f1"),
        "average_char_similarity": deterministic_average("char_similarity"),
        "number_unit_recall": deterministic_average("number_unit_recall"),
        "model_code_recall": deterministic_average("model_code_recall"),
        "url_recall": deterministic_average("url_recall"),
        "generation_latency_seconds": {
            "mean": _mean(generation_times),
            "p50": _percentile(generation_times, 0.50),
            "p95": _percentile(generation_times, 0.95),
        },
        "judge_latency_seconds": {
            "mean": _mean(judge_times),
            "p50": _percentile(judge_times, 0.50),
            "p95": _percentile(judge_times, 0.95),
        },
        "per_source": {
            source: {
                "count": len(items),
                "passed": sum(bool(item["semantic_pass"]) for item in items),
                "accuracy": sum(bool(item["semantic_pass"]) for item in items) / len(items),
            }
            for source, items in sorted(per_source.items())
        },
    }


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            completed[record["name"]] = record
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"[WARN] checkpoint 第 {line_number} 行无效: {exc}", flush=True)
    return completed



def append_checkpoint(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        handle.flush()


def load_answer_cases(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        cases = payload.get("cases")
        manifest = {key: value for key, value in payload.items() if key != "cases"}
    else:
        cases = payload
        manifest = {"case_file": str(path.resolve())}
    if not isinstance(cases, list):
        raise ValueError("评测集必须是列表，或包含 cases 列表的 JSON 对象")
    required = {"name", "query", "reference", "expected_source"}
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"第 {index} 条评测数据不是对象")
        missing = sorted(required - set(case))
        if missing:
            raise ValueError(f"第 {index} 条评测数据缺少字段: {missing}")
    manifest["case_file"] = str(path.resolve())
    return cases, manifest

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate EnBot answer accuracy on new data.")
    parser.add_argument("--api", default="http://127.0.0.1:8000/v1/chat")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--ollama-api", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--judge-model")
    parser.add_argument("--knowledge-root", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit 不能小于 0")

    load_dotenv(PROJECT_ROOT / ".env")
    knowledge_root = args.knowledge_root or Path(
        os.getenv("KNOWLEDGE_FILE", PROJECT_ROOT / "knowledge_data")
    )
    api_key = args.api_key if args.api_key is not None else os.getenv("ENBOT_API_KEY", "")
    judge_model = args.judge_model or os.getenv("AB_JUDGE_MODEL") or os.getenv(
        "OLLAMA_MODEL", "enbot-v9:7b"
    )
    output = args.output.resolve()
    checkpoint = (args.checkpoint or output.with_suffix(".jsonl")).resolve()

    if args.cases:
        answer_cases, manifest = load_answer_cases(args.cases)
    else:
        all_cases, manifest = build_cases(Path(knowledge_root))
        answer_cases = [case for case in all_cases if case.get("reference")]
    cases = answer_cases[: args.limit] if args.limit else answer_cases
    if not cases:
        raise RuntimeError("新知识库中没有可评测的问答标准答案")

    completed = load_checkpoint(checkpoint) if args.resume else {}
    if not args.resume and checkpoint.exists():
        checkpoint.unlink()
    pending = [case for case in cases if case["name"] not in completed]

    print(f"知识库: {Path(knowledge_root).expanduser().resolve()}", flush=True)
    print(f"聊天 API: {args.api}", flush=True)
    print(f"评测用例: {len(cases)}（完整问答 {len(answer_cases)}）", flush=True)
    print(f"裁判模型: {'已关闭' if args.skip_judge else judge_model}", flush=True)
    print(f"已完成: {len(completed)}；待执行: {len(pending)}", flush=True)
    print(f"checkpoint: {checkpoint}", flush=True)

    for case in pending:
        result = evaluate_case(
            case,
            api_url=args.api,
            api_key=api_key,
            ollama_url=args.ollama_api,
            judge_model=judge_model,
            timeout=args.timeout,
            retries=args.retries,
            skip_judge=args.skip_judge,
        )
        completed[result["name"]] = result
        append_checkpoint(checkpoint, result)
        done = sum(name in completed for name in (item["name"] for item in cases))
        if result.get("chat_error"):
            state = "CHAT_ERROR"
            detail = result["chat_error"]
        elif result.get("judge_error"):
            state = "JUDGE_ERROR"
            detail = result["judge_error"]
        elif args.skip_judge:
            state = "GENERATED"
            detail = f"token_f1={result['deterministic']['token_f1']:.4f}"
        else:
            state = "PASS" if result["semantic_pass"] else "FAIL"
            judged = result["judge"]
            detail = (
                f"correctness={judged['correctness']}/5 "
                f"complete={judged['completeness']}/5 "
                f"relevant={judged['relevance']}/5 "
                f"hallucination={judged['hallucination']}"
            )
        print(
            f"[{done}/{len(cases)}] {state} {result['name']} {detail} | "
            f"{result['query'][:80]}",
            flush=True,
        )

    ordered_results = [
        completed[case["name"]] for case in cases if case["name"] in completed
    ]
    summary = summarize(cases, ordered_results)
    report = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": (
            "deterministic_reference_metrics_only"
            if args.skip_judge
            else "local_blind_llm_judge_plus_deterministic_metrics"
        ),
        "candidate_api": args.api,
        "judge_model": None if args.skip_judge else judge_model,
        "pass_rule": (
            "correctness>=4, completeness>=4, relevance>=4, hallucination=false"
            if not args.skip_judge
            else None
        ),
        "manifest": manifest,
        "summary": summary,
        "failures": [
            item
            for item in ordered_results
            if item.get("chat_error")
            or item.get("judge_error")
            or item.get("semantic_pass") is False
        ],
        "results": ordered_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("", flush=True)
    if args.skip_judge:
        print(
            "Summary: "
            f"generated={summary['generated_answers']}/{summary['requested_cases']}, "
            f"token_f1={summary['average_token_f1']:.4f}, "
            f"char_similarity={summary['average_char_similarity']:.4f}",
            flush=True,
        )
    else:
        accuracy = summary["semantic_accuracy"]
        print(
            "Summary: "
            f"judged={summary['judged_answers']}/{summary['requested_cases']}, "
            f"passed={summary['semantic_passed']}, "
            f"accuracy={accuracy:.4f}, "
            f"avg_correctness={summary['average_correctness_1_to_5']:.4f}/5, "
            f"hallucination_rate={summary['hallucination_rate']:.4f}, "
            f"chat_errors={summary['chat_errors']}, "
            f"judge_errors={summary['judge_errors']}",
            flush=True,
        )
    print(f"报告: {output}", flush=True)
    return 2 if summary["chat_errors"] or summary["judge_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())