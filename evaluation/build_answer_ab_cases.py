"""Build the fixed answer-level A/B set from V3 development positives."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path(__file__).with_name("retrieval_cases_v3.json")
DEFAULT_OUTPUT = Path(__file__).with_name("answer_ab_cases_v1.json")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def field_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, default=str).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()

    if "holdout" in str(args.input.resolve()).lower():
        raise ValueError("Holdout data must not be reused for A/B development evaluation.")
    if args.output.exists():
        raise FileExistsError(f"A/B case file already exists: {args.output}")

    load_dotenv(PROJECT_ROOT / ".env")
    knowledge_root = Path(
        os.getenv("KNOWLEDGE_FILE", PROJECT_ROOT / "knowledge_data")
    )
    source_cases = load_json(args.input)
    positives = [case for case in source_cases if not case.get("expect_empty")]
    if len(positives) < args.limit:
        raise RuntimeError(
            f"Only {len(positives)} positive cases are available; {args.limit} requested."
        )

    source_cache: dict[str, list[Any]] = {}
    cases: list[dict[str, Any]] = []
    for index, case in enumerate(positives[: args.limit], start=1):
        source = str(case["expected_source"])
        row_number = int(case["expected_row"])
        if source not in source_cache:
            payload = load_json(knowledge_root / source)
            source_cache[source] = payload if isinstance(payload, list) else [payload]
        rows = source_cache[source]
        if row_number < 1 or row_number > len(rows):
            raise IndexError(f"Invalid row {row_number} for {source}")
        row = rows[row_number - 1]
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object at {source}:{row_number}")
        reference = field_text(row.get("output"))
        if not reference:
            raise ValueError(f"Missing reference answer at {source}:{row_number}")
        cases.append(
            {
                "name": f"ANSWER-AB-{index:03d}",
                "query": field_text(case.get("query")),
                "reference": reference,
                "expected_source": source,
                "expected_row": row_number,
                "acceptable_documents": case.get("acceptable_documents"),
            }
        )

    output = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "purpose": "M3 answer-level no-RAG versus Milvus-RAG development A/B",
            "source_set": str(args.input.resolve()),
            "case_count": len(cases),
            "holdout_used": False,
            "policy": "Development A/B only. Holdout V2 remains closed.",
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Output: {args.output}")
    print(f"Cases: {len(cases)}")
    print("Holdout used: False")
    print("Status: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
