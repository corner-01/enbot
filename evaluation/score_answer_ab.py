"""Score answer A/B generations using hard facts and a blinded local judge."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path(__file__).parent / "answer_ab" / "v1" / "generations.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "answer_ab" / "v1" / "summary.json"

MODEL_CODE_RE = re.compile(
    r"\b(?:HOME-ESS|HESS-HY|CESS-HY|CESS-HV|HC-EV|IESS|HESS-GATEWAY)"
    r"[-A-Z0-9~./]+\b",
    flags=re.IGNORECASE,
)
NUMBER_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9])\d+(?:\.\d+)?\s*(?:%|V|A|W|KW|KWH|HZ|MM|CM|M|S|MS|°C|℃|P/KWH)?",
    flags=re.IGNORECASE,
)
URL_RE = re.compile(r"https?://[^\s\])]+", flags=re.IGNORECASE)
TOKEN_RE = re.compile(r"[A-Za-z0-9._~/-]+|[\u4e00-\u9fff]")

JUDGE_PROMPT = """You are a strict evaluator for a home-energy-storage support system.

Compare the candidate answer against the reference answer. Ignore writing style and wording differences. Evaluate factual correctness, completeness for the user's question, relevance, and unsupported claims. A statement is hallucinated only when it adds a concrete fact that is unsupported by or contradicts the reference.

Return one JSON object only, with this exact schema:
{{"correctness": 1, "completeness": 1, "relevance": 1, "hallucination": false, "reason": "short explanation"}}

Scores are integers from 1 to 5. A score of 4 means essentially correct with only minor omissions. A score of 5 means fully correct.

Question:
{query}

Reference answer:
{reference}

Candidate answer:
{candidate}
"""


def normalize_fact(value: str) -> str:
    return re.sub(r"\s+", "", value).lower().rstrip(".,;:，。；：")


def fact_set(pattern: re.Pattern[str], value: str) -> set[str]:
    return {normalize_fact(match.group(0)) for match in pattern.finditer(value or "")}


def fact_recall(pattern: re.Pattern[str], reference: str, candidate: str) -> float | None:
    expected = fact_set(pattern, reference)
    if not expected:
        return None
    actual = fact_set(pattern, candidate)
    return len(expected & actual) / len(expected)


def tokens(value: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(value or "")]


def token_f1(reference: str, candidate: str) -> float:
    expected = Counter(tokens(reference))
    actual = Counter(tokens(candidate))
    if not expected or not actual:
        return 0.0
    overlap = sum((expected & actual).values())
    precision = overlap / sum(actual.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def post_json(url: str, payload: dict[str, Any], timeout: int = 600) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_judge_json(value: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", value, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Judge did not return JSON: {value[:300]}")
    parsed = json.loads(match.group(0))
    for key in ("correctness", "completeness", "relevance"):
        parsed[key] = int(parsed[key])
        if not 1 <= parsed[key] <= 5:
            raise ValueError(f"Invalid {key}: {parsed[key]}")
    hallucination = parsed["hallucination"]
    if isinstance(hallucination, str):
        normalized = hallucination.strip().lower()
        if normalized not in {"true", "false"}:
            raise ValueError(f"Invalid hallucination value: {hallucination}")
        hallucination = normalized == "true"
    parsed["hallucination"] = bool(hallucination)
    parsed["reason"] = str(parsed.get("reason", ""))
    return parsed


def judge(
    ollama_url: str,
    model: str,
    query: str,
    reference: str,
    candidate: str,
) -> dict[str, Any]:
    prompt = JUDGE_PROMPT.format(
        query=query,
        reference=reference,
        candidate=candidate,
    )
    response = post_json(
        ollama_url,
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "keep_alive": "30m",
            "options": {"temperature": 0, "seed": 314159, "num_predict": 256},
        },
    )
    return parse_judge_json(str(response.get("response", "")))


def deterministic_scores(reference: str, candidate: str) -> dict[str, Any]:
    return {
        "token_f1": token_f1(reference, candidate),
        "char_similarity": SequenceMatcher(None, reference, candidate).ratio(),
        "number_unit_recall": fact_recall(NUMBER_UNIT_RE, reference, candidate),
        "model_code_recall": fact_recall(MODEL_CODE_RE, reference, candidate),
        "url_recall": fact_recall(URL_RE, reference, candidate),
    }


def mean_present(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return statistics.mean(values) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "semantic_pass_rate": statistics.mean(row["semantic_pass"] for row in rows),
        "average_correctness": statistics.mean(row["correctness"] for row in rows),
        "average_completeness": statistics.mean(row["completeness"] for row in rows),
        "average_relevance": statistics.mean(row["relevance"] for row in rows),
        "hallucination_rate": statistics.mean(row["hallucination"] for row in rows),
        "average_token_f1": statistics.mean(row["token_f1"] for row in rows),
        "average_char_similarity": statistics.mean(row["char_similarity"] for row in rows),
        "number_unit_recall": mean_present(rows, "number_unit_recall"),
        "model_code_recall": mean_present(rows, "model_code_recall"),
        "url_recall": mean_present(rows, "url_recall"),
        "average_generation_seconds": statistics.mean(
            row["generation_seconds"] for row in rows
        ),
        "average_tokens_per_second": mean_present(rows, "tokens_per_second"),
    }


def clean_float(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, 6)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ollama-api", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--judge-model")
    args = parser.parse_args()

    if "holdout" in str(args.input.resolve()).lower():
        raise ValueError("Holdout files are forbidden in this development A/B score run.")
    if args.output.exists():
        raise FileExistsError(f"A/B summary already exists: {args.output}")

    load_dotenv(PROJECT_ROOT / ".env")
    judge_model = args.judge_model or os.getenv("AB_JUDGE_MODEL") or os.getenv(
        "OLLAMA_MODEL", "enbot-optuna:7b"
    )
    generations = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not generations:
        raise ValueError("Generation file is empty")

    scored: dict[str, list[dict[str, Any]]] = {"baseline": [], "rag": []}
    case_results: list[dict[str, Any]] = []
    parse_failures = 0
    for index, generation in enumerate(generations, start=1):
        case_result = {"name": generation["name"], "arms": {}}
        for arm in ("baseline", "rag"):
            answer = generation[arm]["answer"]
            deterministic = deterministic_scores(generation["reference"], answer)
            try:
                judged = judge(
                    args.ollama_api,
                    judge_model,
                    generation["query"],
                    generation["reference"],
                    answer,
                )
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                parse_failures += 1
                judged = {
                    "correctness": 0,
                    "completeness": 0,
                    "relevance": 0,
                    "hallucination": True,
                    "reason": f"judge_parse_failure: {exc}",
                }
            semantic_pass = (
                judged["correctness"] >= 4
                and judged["completeness"] >= 4
                and judged["relevance"] >= 4
                and not judged["hallucination"]
            )
            arm_result = {
                **deterministic,
                **judged,
                "semantic_pass": semantic_pass,
                "generation_seconds": generation[arm]["wall_seconds"],
                "tokens_per_second": generation[arm].get("tokens_per_second"),
            }
            scored[arm].append(arm_result)
            case_result["arms"][arm] = arm_result
        case_results.append(case_result)
        print(f"[{index}/{len(generations)}] {generation['name']} judged")

    baseline_summary = summarize(scored["baseline"])
    rag_summary = summarize(scored["rag"])
    pass_rate_delta = (
        rag_summary["semantic_pass_rate"] - baseline_summary["semantic_pass_rate"]
    )
    hallucination_delta = (
        rag_summary["hallucination_rate"] - baseline_summary["hallucination_rate"]
    )
    checks = {
        "semantic_pass_rate_improvement_ge_10pp": pass_rate_delta >= 0.10,
        "average_correctness_not_lower": (
            rag_summary["average_correctness"]
            >= baseline_summary["average_correctness"]
        ),
        "hallucination_rate_not_higher": hallucination_delta <= 0.0,
    }
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "mode": "same_model_same_prompt_local_blind_judge",
        "judge_model": judge_model,
        "input": str(args.input.resolve()),
        "case_count": len(generations),
        "judge_parse_failures": parse_failures,
        "baseline": baseline_summary,
        "rag": rag_summary,
        "delta": {
            "semantic_pass_rate": pass_rate_delta,
            "hallucination_rate": hallucination_delta,
            "average_correctness": (
                rag_summary["average_correctness"]
                - baseline_summary["average_correctness"]
            ),
            "average_completeness": (
                rag_summary["average_completeness"]
                - baseline_summary["average_completeness"]
            ),
        },
        "checks": checks,
        "case_results": case_results,
    }

    def normalize_floats(value: Any) -> Any:
        if isinstance(value, float):
            return clean_float(value)
        if isinstance(value, list):
            return [normalize_floats(item) for item in value]
        if isinstance(value, dict):
            return {key: normalize_floats(item) for key, item in value.items()}
        return value

    output = normalize_floats(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nA/B summary")
    print(json.dumps({key: output[key] for key in ("status", "baseline", "rag", "delta", "checks")}, ensure_ascii=False, indent=2))
    print(f"Output: {args.output}")
    return 0 if output["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
