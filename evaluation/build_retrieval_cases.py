"""Build a deterministic, leakage-safe retrieval development set V3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = EVALUATION_DIR / "retrieval_cases_v3.json"
DEFAULT_EXCLUDES = [
    EVALUATION_DIR / "retrieval_cases_v2.json",
    EVALUATION_DIR / "holdout" / "retrieval_holdout_v1.json",
]

NEGATIVE_QUERIES = [
    "推荐三本适合大学生阅读的推理小说。",
    "上海明天的天气怎么样？",
    "Excel中VLOOKUP函数应该怎么使用？",
    "红烧肉的家常做法是什么？",
    "下个月比特币价格会涨到多少？",
    "帮我写一篇英语自我介绍。",
    "把“欢迎来到中国”翻译成法语。",
    "不定积分的基本计算方法是什么？",
    "足球比赛中的越位规则是什么？",
    "打印机卡纸以后应该怎么处理？",
    "XESS-HY-Q-900K支持量子电网通信吗？",
    "HESS-HY-Z-505K出现F.888故障是什么意思？",
    "HOME-ESS-HV-123K的安装间距是多少？",
    "CESS-HY-777K是否通过国际船舶认证？",
    "EnValt Quantum App如何预测未来十年的电价？",
    "HESS-Gateway-X9如何连接卫星网络？",
    "HC-EV-AC-99K是否支持无线充电？",
    "HOME-ESS-LV-42K采用了什么核聚变材料？",
    "iESS-M99是否支持深海环境运行？",
    "SolarWizard逆变器如何连接EnValt系统？",
]

EQUIVALENT_DOCUMENTS = {
    ("HOME-ESS-LV-3.68K.json", 29): [
        {"source": "HOME-ESS-LV-3.68K.json", "row": 29},
        {"source": "HOME-ESS-LV-3.68K (1).json", "row": 29},
    ],
    ("场景化问答.json", 171): [
        {"source": "场景化问答.json", "row": 171},
        {"source": "场景化问答(1).json", "row": 171},
    ],
    ("垂直领域.json", 3): [
        {"source": "垂直领域.json", "row": 3},
        {"source": "垂直领域(1).json", "row": 3},
    ],
}


def field_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, default=str).strip()


def normalize_query(value: str) -> str:
    return " ".join(value.lower().split())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_exclusions(
    paths: list[Path],
) -> tuple[set[tuple[str, int]], set[str]]:
    excluded_rows: set[tuple[str, int]] = set()
    excluded_queries: set[str] = set()

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"排除数据集不存在: {path}")
        cases = load_json(path)
        if not isinstance(cases, list):
            raise ValueError(f"排除数据集必须是JSON数组: {path}")

        for case in cases:
            if not isinstance(case, dict):
                continue
            source = case.get("expected_source")
            row_number = case.get("expected_row")
            if source and row_number is not None:
                excluded_rows.add((str(source), int(row_number)))
            query = field_text(case.get("query"))
            if query:
                excluded_queries.add(normalize_query(query))

    return excluded_rows, excluded_queries


def is_atomic_query(query: str) -> bool:
    """Keep development examples close to a normal single user request."""
    if len(query) < 8 or len(query) > 300:
        return False
    numbered_questions = re.findall(r"(?m)^\s*\d+\s*[.)、]\s*", query)
    if len(numbered_questions) >= 3:
        return False
    question_marks = query.count("?") + query.count("？")
    return question_marks <= 2


def load_candidates(
    root: Path,
    excluded_rows: set[tuple[str, int]],
    excluded_queries: set[str],
) -> dict[str, deque[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_queries = set(excluded_queries)

    for path in sorted(root.rglob("*.json")):
        payload = load_json(path)
        rows = payload if isinstance(payload, list) else [payload]
        source = path.relative_to(root).as_posix()

        for row_number, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            if (source, row_number) in excluded_rows:
                continue

            instruction = field_text(row.get("instruction"))
            input_text = field_text(row.get("input"))
            answer = field_text(row.get("output"))
            query = "\n".join(
                part for part in (instruction, input_text) if part
            ).strip()
            normalized = normalize_query(query)

            if not answer or not is_atomic_query(query):
                continue
            if normalized in seen_queries:
                continue

            seen_queries.add(normalized)
            candidates[source].append(
                {
                    "query": query,
                    "expected_source": source,
                    "expected_row": row_number,
                }
            )

    output: dict[str, deque[dict[str, Any]]] = {}
    for source, rows in candidates.items():
        rows.sort(
            key=lambda case: hashlib.sha256(
                f"{source}:{case['expected_row']}:enbot-dev-v3".encode("utf-8")
            ).hexdigest()
        )
        output[source] = deque(rows)
    return output


def balanced_sample(
    candidates: dict[str, deque[dict[str, Any]]], count: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    active_sources = deque(sorted(candidates))

    while active_sources and len(selected) < count:
        source = active_sources.popleft()
        rows = candidates[source]
        if rows:
            case = rows.popleft()
            case["name"] = f"DEV3-POS-{len(selected) + 1:03d}"
            selected.append(case)
        if rows:
            active_sources.append(source)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positives", type=int, default=60)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--exclude",
        action="append",
        type=Path,
        help="需要排除的数据集路径；可以多次指定",
    )
    args = parser.parse_args()

    if args.positives < 1:
        raise ValueError("正例数量必须大于0")

    load_dotenv(PROJECT_ROOT / ".env")
    knowledge_root = Path(
        os.getenv("KNOWLEDGE_FILE", PROJECT_ROOT / "knowledge_data")
    )
    if not knowledge_root.exists():
        raise FileNotFoundError(f"知识库目录不存在: {knowledge_root}")

    exclude_paths = args.exclude or DEFAULT_EXCLUDES
    excluded_rows, excluded_queries = load_exclusions(exclude_paths)
    candidates = load_candidates(
        knowledge_root, excluded_rows, excluded_queries
    )
    positives = balanced_sample(candidates, args.positives)
    if len(positives) < args.positives:
        raise RuntimeError(
            f"只找到 {len(positives)} 条可用正例，要求 {args.positives} 条"
        )

    for case in positives:
        key = (case["expected_source"], case["expected_row"])
        if key in EQUIVALENT_DOCUMENTS:
            case["acceptable_documents"] = EQUIVALENT_DOCUMENTS[key]
            case["label_note"] = "Both source files contain the same Q&A record."

    negatives = [
        {
            "name": f"DEV3-NEG-{index:03d}",
            "query": query,
            "expect_empty": True,
        }
        for index, query in enumerate(NEGATIVE_QUERIES, start=1)
    ]
    cases = positives + negatives
    positive_rows = {
        (case["expected_source"], case["expected_row"])
        for case in positives
    }
    overlap = positive_rows & excluded_rows
    if overlap:
        raise RuntimeError(f"检测到数据泄漏: {sorted(overlap)}")

    all_queries = [normalize_query(case["query"]) for case in cases]
    if len(all_queries) != len(set(all_queries)):
        raise RuntimeError("V3中存在重复问题")
    if any(query in excluded_queries for query in all_queries):
        raise RuntimeError("V3中存在与排除集相同的问题")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"输出文件: {args.output}")
    print(f"正例数量: {len(positives)}")
    print(f"负例数量: {len(negatives)}")
    print(f"总数量: {len(cases)}")
    print(f"与排除集重叠: {len(overlap)}")
    print("状态: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
