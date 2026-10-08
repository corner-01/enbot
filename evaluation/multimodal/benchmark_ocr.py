"""Measure cold-load and warm-inference latency for the configured OCR engine."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.vision.ocr_engine import PaddleOCREngine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default=str(
            PROJECT_ROOT
            / "evaluation"
            / "multimodal"
            / "images"
            / "wiring-smoke-test.png"
        ),
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT
            / "evaluation"
            / "multimodal"
            / "results"
            / "mobile-warm-benchmark.json"
        ),
    )
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2 to measure warm inference")

    engine = PaddleOCREngine()
    runs: list[dict[str, object]] = []
    last_result = None
    for run_number in range(1, args.runs + 1):
        result = engine.extract(args.image)
        last_result = result
        record = {
            "run": run_number,
            "inference_ms": round(result.inference_ms, 3),
            "block_count": len(result.blocks),
            "discarded_blocks": result.discarded_blocks,
            "measurements": [item.normalized for item in result.measurements],
        }
        runs.append(record)
        print(json.dumps(record, ensure_ascii=False))

    assert last_result is not None
    warm_values = [float(item["inference_ms"]) for item in runs[1:]]
    summary = {
        "image": str(Path(args.image).resolve()),
        "engine": last_result.engine,
        "detection_model": last_result.detection_model,
        "recognition_model": last_result.recognition_model,
        "model_load_ms": round(last_result.model_load_ms, 3),
        "cold_inference_ms": runs[0]["inference_ms"],
        "warm_inference_average_ms": round(statistics.mean(warm_values), 3),
        "runs": runs,
        "blocks": [asdict(item) for item in last_result.blocks],
        "measurements": [asdict(item) for item in last_result.measurements],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
