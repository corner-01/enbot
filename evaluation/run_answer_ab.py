"""Generate blind no-RAG and RAG answers with the same local Ollama model."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = Path(__file__).with_name("answer_ab_cases_v1.json")
DEFAULT_OUTPUT = Path(__file__).parent / "answer_ab" / "v1" / "generations.jsonl"

PROMPT_TEMPLATE = """You are Xiaoying, a professional home-energy-storage support assistant.

Answer the user's question accurately and directly. Use the supplied knowledge context when it is available. If no context is supplied, answer from your learned knowledge. Do not invent product models, fault codes, numbers, units, URLs, procedures, or causes. If the available information is insufficient, clearly say that it is not recorded or cannot be confirmed. Answer in the same language as the user.

User question:
{query}

Knowledge context:
{context}

Final answer:"""


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


def retrieve(api_url: str, query: str) -> list[dict[str, Any]]:
    response = post_json(
        api_url,
        {"query": query, "include_content": True},
        timeout=300,
    )
    if response.get("status") != "success":
        raise RuntimeError(f"Retrieval failed: {response}")
    return response.get("data", {}).get("retrieval_debug", [])


def generate(
    ollama_url: str,
    model: str,
    query: str,
    context: str,
) -> dict[str, Any]:
    prompt = PROMPT_TEMPLATE.format(
        query=query,
        context=context or "[No external knowledge context was supplied.]",
    )
    started = time.perf_counter()
    response = post_json(
        ollama_url,
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0,
                "seed": 42,
                "num_predict": 512,
            },
        },
    )
    wall_seconds = time.perf_counter() - started
    answer = str(response.get("response", "")).strip()
    if not answer:
        raise RuntimeError(f"Ollama returned an empty answer: {response}")
    eval_count = int(response.get("eval_count") or 0)
    eval_duration = int(response.get("eval_duration") or 0)
    tokens_per_second = (
        eval_count / (eval_duration / 1_000_000_000)
        if eval_count and eval_duration
        else None
    )
    return {
        "answer": answer,
        "wall_seconds": wall_seconds,
        "total_duration_seconds": (response.get("total_duration") or 0) / 1_000_000_000,
        "prompt_eval_duration_seconds": (
            (response.get("prompt_eval_duration") or 0) / 1_000_000_000
        ),
        "eval_duration_seconds": eval_duration / 1_000_000_000,
        "eval_count": eval_count,
        "tokens_per_second": tokens_per_second,
    }


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            completed.add(json.loads(line)["name"])
    return completed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--retrieve-api", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--ollama-api", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--model")
    args = parser.parse_args()

    if "holdout" in str(args.cases.resolve()).lower():
        raise ValueError("Holdout files are forbidden in this development A/B run.")

    load_dotenv(PROJECT_ROOT / ".env")
    model = args.model or os.getenv("OLLAMA_MODEL", "enbot-optuna:7b")
    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = payload["cases"]
    completed = load_completed(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for index, case in enumerate(cases, start=1):
        if case["name"] in completed:
            print(f"[SKIP] {case['name']} already completed")
            continue

        documents = retrieve(args.retrieve_api, case["query"])
        contexts = [
            str(document.get("content") or document.get("preview") or "").strip()
            for document in documents
            if str(document.get("content") or document.get("preview") or "").strip()
        ]
        rag_context = "\n\n".join(
            f"[Context {context_index}]\n{context}"
            for context_index, context in enumerate(contexts, start=1)
        )

        # Alternate generation order so warm caches do not always favor one arm.
        if index % 2:
            baseline = generate(args.ollama_api, model, case["query"], "")
            rag = generate(args.ollama_api, model, case["query"], rag_context)
        else:
            rag = generate(args.ollama_api, model, case["query"], rag_context)
            baseline = generate(args.ollama_api, model, case["query"], "")

        record = {
            "name": case["name"],
            "query": case["query"],
            "reference": case["reference"],
            "expected_source": case["expected_source"],
            "expected_row": case["expected_row"],
            "model": model,
            "retrieved": [
                {
                    "rank": document.get("rank"),
                    "source": document.get("source"),
                    "row": document.get("row"),
                    "reranker_score": document.get("reranker_score"),
                }
                for document in documents
            ],
            "baseline": baseline,
            "rag": rag,
        }
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"[{index}/{len(cases)}] {case['name']} saved; "
            f"baseline={baseline['wall_seconds']:.2f}s, "
            f"rag={rag['wall_seconds']:.2f}s"
        )

    print(f"Output: {args.output}")
    print(f"Completed: {len(load_completed(args.output))}/{len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
