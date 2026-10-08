"""Build a locked retrieval Holdout V2 from previously unused knowledge rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = EVALUATION_DIR / "holdout" / "retrieval_holdout_v2.json"
DEFAULT_EXCLUDES = [
    EVALUATION_DIR / "retrieval_cases_v2.json",
    EVALUATION_DIR / "retrieval_cases_v3.json",
    EVALUATION_DIR / "holdout" / "retrieval_holdout_v1.json",
]
SELECTION_SALT = "enbot-retrieval-holdout-v2-2026-09-29"

NEGATIVE_QUERIES = [
    "请推荐三本适合周末阅读的科幻小说。",
    "上海今天的天气和空气质量怎么样？",
    "Excel 的 INDEX 和 MATCH 函数怎么组合使用？",
    "感冒发烧后应该吃什么药？",
    "请预测下个月比特币的最高价格。",
    "帮我写一篇大学生英文自我介绍。",
    "把‘项目进展顺利’翻译成法语。",
    "如何根据半径和高度计算圆柱体的体积？",
    "汽车发动机出现爆震应该怎么维修？",
    "打印机一直卡纸该怎么处理？",
    "XESS-HY-Q-901K 支持哪些并网标准？",
    "HESS-HY-Z-606K 的故障码 F.777 表示什么？",
    "HOME-ESS-HV-124K 的安装尺寸是多少？",
    "CESS-HY-778K 是否通过火星电网认证？",
    "EnValt Quantum App 如何预测未来三十年的电价？",
    "HESS-Gateway-X10 能否连接卫星互联网？",
    "HC-EV-AC-100K 是否支持无线充电？",
    "HOME-ESS-LV-43K 使用什么超导电池材料？",
    "iESS-M100 是否支持海底储能系统？",
    "SolarWizard 机器人怎样自动安装储能系统？",
]


def field_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, default=str).strip()


def normalize(value: str) -> str:
    return " ".join(value.lower().split())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_atomic_query(query: str) -> bool:
    if len(query) < 8 or len(query) > 300:
        return False
    numbered = re.findall(r"(?m)^\s*\d+\s*[.)、]\s*", query)
    question_marks = query.count("?") + query.count("？")
    return len(numbered) < 3 and question_marks <= 2


def load_exclusions(paths: list[Path]) -> tuple[set[tuple[str, int]], set[str]]:
    rows: set[tuple[str, int]] = set()
    queries: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Exclusion set does not exist: {path}")
        payload = load_json(path)
        if not isinstance(payload, list):
            raise ValueError(f"Exclusion set is not a JSON list: {path}")
        for case in payload:
            if not isinstance(case, dict):
                continue
            documents = case.get("acceptable_documents") or [
                {
                    "source": case.get("expected_source"),
                    "row": case.get("expected_row"),
                }
            ]
            for document in documents:
                source = document.get("source")
                row = document.get("row")
                if source and row is not None:
                    rows.add((str(source), int(row)))
            query = field_text(case.get("query"))
            if query:
                queries.add(normalize(query))
    return rows, queries


def load_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        payload = load_json(path)
        rows = payload if isinstance(payload, list) else [payload]
        source = path.relative_to(root).as_posix()
        for row_number, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            instruction = field_text(row.get("instruction"))
            input_text = field_text(row.get("input"))
            answer = field_text(row.get("output"))
            query = "\n".join(
                part for part in (instruction, input_text) if part
            ).strip()
            if not answer or not is_atomic_query(query):
                continue
            signature_text = f"{normalize(query)}\n{normalize(answer)}"
            signature = hashlib.sha256(signature_text.encode("utf-8")).hexdigest()
            records.append(
                {
                    "source": source,
                    "row": row_number,
                    "query": query,
                    "signature": signature,
                }
            )
    return records


def build_candidates(
    records: list[dict[str, Any]],
    excluded_rows: set[tuple[str, int]],
    excluded_queries: set[str],
) -> dict[str, deque[dict[str, Any]]]:
    by_signature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_signature[record["signature"]].append(record)

    excluded_signatures = {
        record["signature"]
        for record in records
        if (record["source"], record["row"]) in excluded_rows
    }
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_queries = set(excluded_queries)

    for signature, equivalents in sorted(by_signature.items()):
        if signature in excluded_signatures:
            continue
        equivalents.sort(key=lambda item: (item["source"], item["row"]))
        canonical = equivalents[0]
        normalized_query = normalize(canonical["query"])
        if normalized_query in seen_queries:
            continue
        seen_queries.add(normalized_query)
        case: dict[str, Any] = {
            "query": canonical["query"],
            "expected_source": canonical["source"],
            "expected_row": canonical["row"],
        }
        if len(equivalents) > 1:
            case["acceptable_documents"] = [
                {"source": item["source"], "row": item["row"]}
                for item in equivalents
            ]
            case["label_note"] = "Equivalent Q&A records share the same content."
        candidates[canonical["source"]].append(case)

    output: dict[str, deque[dict[str, Any]]] = {}
    for source, cases in candidates.items():
        cases.sort(
            key=lambda case: hashlib.sha256(
                (
                    f"{SELECTION_SALT}:{source}:"
                    f"{case['expected_row']}:{normalize(case['query'])}"
                ).encode("utf-8")
            ).hexdigest()
        )
        output[source] = deque(cases)
    return output


def balanced_sample(
    candidates: dict[str, deque[dict[str, Any]]], count: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    active = deque(sorted(candidates))
    while active and len(selected) < count:
        source = active.popleft()
        source_cases = candidates[source]
        if source_cases:
            selected.append(source_cases.popleft())
        if source_cases:
            active.append(source)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positives", type=int, default=40)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--exclude", action="append", type=Path)
    args = parser.parse_args()

    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        raise FileExistsError(
            f"Holdout V2 is already locked: {args.output}. Do not overwrite it."
        )
    if args.positives < 1:
        raise ValueError("--positives must be greater than zero")

    load_dotenv(PROJECT_ROOT / ".env")
    knowledge_root = Path(
        os.getenv("KNOWLEDGE_FILE", PROJECT_ROOT / "knowledge_data")
    )
    if not knowledge_root.exists():
        raise FileNotFoundError(f"Knowledge directory does not exist: {knowledge_root}")

    exclude_paths = args.exclude or DEFAULT_EXCLUDES
    excluded_rows, excluded_queries = load_exclusions(exclude_paths)
    records = load_records(knowledge_root)
    candidates = build_candidates(records, excluded_rows, excluded_queries)
    positives = balanced_sample(candidates, args.positives)
    if len(positives) != args.positives:
        raise RuntimeError(
            f"Only {len(positives)} unused positive records were available; "
            f"{args.positives} were requested."
        )

    for index, case in enumerate(positives, start=1):
        case["name"] = f"HOLDOUT2-POS-{index:03d}"
    negatives = [
        {
            "name": f"HOLDOUT2-NEG-{index:03d}",
            "query": query,
            "expect_empty": True,
        }
        for index, query in enumerate(NEGATIVE_QUERIES, start=1)
    ]
    cases = positives + negatives

    selected_rows = {
        (case["expected_source"], case["expected_row"]) for case in positives
    }
    overlap = selected_rows & excluded_rows
    all_queries = [normalize(case["query"]) for case in cases]
    if overlap:
        raise RuntimeError(f"Holdout V2 row leakage detected: {sorted(overlap)}")
    if len(all_queries) != len(set(all_queries)):
        raise RuntimeError("Holdout V2 contains duplicate queries")
    if any(query in excluded_queries for query in all_queries):
        raise RuntimeError("Holdout V2 query leakage detected")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "LOCKED_UNEVALUATED",
        "generator": "deterministic_source_balanced_unseen_records",
        "selection_salt": SELECTION_SALT,
        "positive_cases": len(positives),
        "negative_cases": len(negatives),
        "excluded_sets": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in exclude_paths
        ],
        "overlap_rows": len(overlap),
        "sha256": sha256_file(args.output),
        "policy": (
            "Evaluate this frozen candidate once. Do not tune retrieval rules "
            "or parameters against this holdout."
        ),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Locked holdout: {args.output}")
    print(f"Manifest: {manifest_path}")
    print(f"SHA256: {manifest['sha256']}")
    print(f"Cases: {len(positives)} positive + {len(negatives)} negative")
    print("Overlap with V2/V3/Holdout V1: 0")
    print("Status: LOCKED_UNEVALUATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
