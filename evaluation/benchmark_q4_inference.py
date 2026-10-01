"""Benchmark Q4 Ollama inference and end-to-end Milvus RAG latency."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.run_answer_ab import PROMPT_TEMPLATE, post_json, retrieve


DEFAULT_CASES = Path(__file__).with_name("answer_ab_cases_v1.json")
DEFAULT_OUTPUT = Path(__file__).parent / "performance" / "q4_inference_v1.json"
THRESHOLDS = {
    "ttft_p95_seconds": 2.0,
    "end_to_end_p50_seconds": 3.0,
    "end_to_end_p95_seconds": 5.0,
    "average_tokens_per_second": 20.0,
}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile from an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def gpu_memory_used_mib() -> float | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        first_line = completed.stdout.strip().splitlines()[0]
        return float(first_line.strip())
    except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
        return None


class GPUMemoryMonitor:
    def __init__(self, interval_seconds: float = 0.25):
        self.interval_seconds = interval_seconds
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "GPUMemoryMonitor":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            value = gpu_memory_used_mib()
            if value is not None:
                self.samples.append(value)
            self._stop.wait(self.interval_seconds)

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    @property
    def peak_mib(self) -> float | None:
        return max(self.samples) if self.samples else None


def stream_generate(
    ollama_url: str,
    model: str,
    prompt: str,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "keep_alive": "30m",
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_predict": 512,
        },
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        ollama_url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    final: dict[str, Any] = {}
    chunks: list[str] = []
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw_line in response:
            if not raw_line.strip():
                continue
            item = json.loads(raw_line.decode("utf-8"))
            chunk = str(item.get("response", ""))
            if chunk:
                chunks.append(chunk)
                if first_token_at is None:
                    first_token_at = time.perf_counter()
            if item.get("done"):
                final = item
    finished = time.perf_counter()
    eval_count = int(final.get("eval_count") or 0)
    eval_duration = int(final.get("eval_duration") or 0)
    return {
        "answer": "".join(chunks).strip(),
        "ttft_seconds": (
            first_token_at - started if first_token_at is not None else None
        ),
        "generation_wall_seconds": finished - started,
        "eval_count": eval_count,
        "tokens_per_second": (
            eval_count / (eval_duration / 1_000_000_000)
            if eval_count and eval_duration
            else None
        ),
    }


def build_context(documents: list[dict[str, Any]]) -> str:
    contexts = [
        str(document.get("content") or document.get("preview") or "").strip()
        for document in documents
        if str(document.get("content") or document.get("preview") or "").strip()
    ]
    return "\n\n".join(
        f"[Context {index}]\n{context}"
        for index, context in enumerate(contexts, start=1)
    )


def model_details(ollama_show_url: str, model: str) -> dict[str, Any]:
    response = post_json(ollama_show_url, {"name": model}, timeout=60)
    details = response.get("details") or {}
    return {
        "family": details.get("family"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--retrieve-api", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--ollama-api", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--ollama-show-api", default="http://127.0.0.1:11434/api/show")
    parser.add_argument("--model")
    args = parser.parse_args()

    if "holdout" in str(args.cases.resolve()).lower():
        raise ValueError("Holdout files are forbidden in performance development tests.")
    if args.output.exists():
        raise FileExistsError(f"Performance result already exists: {args.output}")

    load_dotenv(PROJECT_ROOT / ".env")
    model = args.model or os.getenv("OLLAMA_MODEL", "enbot-optuna:7b")
    case_payload = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = case_payload["cases"]
    if args.samples < 1 or args.samples > len(cases):
        raise ValueError(f"--samples must be between 1 and {len(cases)}")
    if args.warmup < 0 or args.warmup > args.samples:
        raise ValueError("--warmup must be between 0 and --samples")
    benchmark_cases = cases[: args.samples]
    details = model_details(args.ollama_show_api, model)

    print(f"Model: {model}")
    print(f"Quantization: {details.get('quantization_level')}")
    print(f"Warmup: {args.warmup}, measured samples: {args.samples}")

    for index, case in enumerate(benchmark_cases[: args.warmup], start=1):
        documents = retrieve(args.retrieve_api, case["query"])
        prompt = PROMPT_TEMPLATE.format(
            query=case["query"],
            context=build_context(documents),
        )
        stream_generate(args.ollama_api, model, prompt)
        print(f"[WARMUP {index}/{args.warmup}] complete")

    idle_gpu_mib = gpu_memory_used_mib()
    results: list[dict[str, Any]] = []
    for index, case in enumerate(benchmark_cases, start=1):
        with GPUMemoryMonitor() as gpu_monitor:
            overall_started = time.perf_counter()
            retrieval_started = time.perf_counter()
            documents = retrieve(args.retrieve_api, case["query"])
            retrieval_seconds = time.perf_counter() - retrieval_started
            context = build_context(documents)
            prompt = PROMPT_TEMPLATE.format(
                query=case["query"],
                context=context,
            )
            generation = stream_generate(args.ollama_api, model, prompt)
            end_to_end_seconds = time.perf_counter() - overall_started

        if not generation["answer"]:
            raise RuntimeError(f"Empty model response for {case['name']}")
        result = {
            "name": case["name"],
            "retrieval_seconds": retrieval_seconds,
            "ttft_seconds": generation["ttft_seconds"],
            "generation_seconds": generation["generation_wall_seconds"],
            "end_to_end_seconds": end_to_end_seconds,
            "output_tokens": generation["eval_count"],
            "tokens_per_second": generation["tokens_per_second"],
            "gpu_peak_mib": gpu_monitor.peak_mib,
            "retrieved_documents": len(documents),
        }
        results.append(result)
        print(
            f"[{index}/{len(benchmark_cases)}] {case['name']}: "
            f"retrieval={retrieval_seconds:.3f}s, "
            f"ttft={generation['ttft_seconds']:.3f}s, "
            f"e2e={end_to_end_seconds:.3f}s, "
            f"tok/s={generation['tokens_per_second']:.2f}"
        )

    ttft = [float(row["ttft_seconds"]) for row in results]
    retrieval_times = [float(row["retrieval_seconds"]) for row in results]
    generation_times = [float(row["generation_seconds"]) for row in results]
    end_to_end = [float(row["end_to_end_seconds"]) for row in results]
    token_rates = [
        float(row["tokens_per_second"])
        for row in results
        if row["tokens_per_second"] is not None
    ]
    gpu_peaks = [
        float(row["gpu_peak_mib"])
        for row in results
        if row["gpu_peak_mib"] is not None
    ]
    summary = {
        "retrieval_average_seconds": statistics.mean(retrieval_times),
        "retrieval_p95_seconds": percentile(retrieval_times, 0.95),
        "ttft_p50_seconds": percentile(ttft, 0.50),
        "ttft_p95_seconds": percentile(ttft, 0.95),
        "generation_average_seconds": statistics.mean(generation_times),
        "generation_p95_seconds": percentile(generation_times, 0.95),
        "end_to_end_p50_seconds": percentile(end_to_end, 0.50),
        "end_to_end_p95_seconds": percentile(end_to_end, 0.95),
        "average_tokens_per_second": statistics.mean(token_rates),
        "gpu_idle_mib": idle_gpu_mib,
        "gpu_peak_mib": max(gpu_peaks) if gpu_peaks else None,
    }
    quantization = str(details.get("quantization_level") or "")
    checks = {
        "model_is_q4": quantization.upper().startswith("Q4"),
        "ttft_p95_le_2s": summary["ttft_p95_seconds"] <= THRESHOLDS["ttft_p95_seconds"],
        "end_to_end_p50_le_3s": (
            summary["end_to_end_p50_seconds"]
            <= THRESHOLDS["end_to_end_p50_seconds"]
        ),
        "end_to_end_p95_le_5s": (
            summary["end_to_end_p95_seconds"]
            <= THRESHOLDS["end_to_end_p95_seconds"]
        ),
        "average_tokens_per_second_ge_20": (
            summary["average_tokens_per_second"]
            >= THRESHOLDS["average_tokens_per_second"]
        ),
    }
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "model": model,
        "model_details": details,
        "samples": len(results),
        "warmup_samples": args.warmup,
        "thresholds": THRESHOLDS,
        "summary": summary,
        "checks": checks,
        "case_results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nQ4 performance summary")
    print(
        json.dumps(
            {key: output[key] for key in ("status", "model_details", "summary", "checks")},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Output: {args.output}")
    return 0 if output["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
