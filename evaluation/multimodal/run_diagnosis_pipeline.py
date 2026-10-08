"""Run the EnBot multimodal diagnosis pipeline for one local image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.vision.diagnosis_pipeline import MultimodalDiagnosisPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Absolute or project-relative image path")
    parser.add_argument(
        "--question",
        default="Please analyze the device status or issue shown in this image.",
    )
    parser.add_argument("--session-id", default="vision-cli")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    with MultimodalDiagnosisPipeline() as pipeline:
        result = pipeline.run(
            args.image,
            question=args.question,
            session_id=args.session_id,
        )
    payload = result.to_dict(include_debug=args.debug)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    print(serialized)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
        print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
