"""Build the retrieval development set for the V2 XLSX/TXT knowledge base."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = EVALUATION_DIR / "retrieval_cases_v4.json"
DEFAULT_MANIFEST = EVALUATION_DIR / "retrieval_cases_v4_manifest.json"

QUESTION_COLUMNS = {"问题", "question", "q"}
ANSWER_COLUMNS = {"答案", "回答", "answer", "a"}

NEGATIVE_QUERIES = [
    "推荐三部适合周末观看的科幻电影。",
    "北京故宫明天几点开门？",
    "Excel 中 VLOOKUP 函数应该怎么使用？",
    "番茄炒蛋的家常做法是什么？",
    "明年的黄金价格会涨到多少？",
    "帮我写一个 Java 冒泡排序程序。",
    "把“绿色能源”翻译成德语。",
    "如何计算一个三角形的内角和？",
    "羽毛球比赛采用什么计分规则？",
    "扫描仪无法识别驱动时应该怎么处理？",
    "XESS-HY-Q-900K 支持量子电网通信吗？",
    "FX-999K-SL 出现 F.888 故障是什么意思？",
    "EBS-H-123K 的安装间距是多少？",
    "EnValt Quantum App 如何预测未来十年的电价？",
    "EPortal-X9 如何连接卫星网络？",
    "EC-99K 是否支持无线充电？",
    "EB-L-42K 采用了什么核聚变材料？",
    "iEnOS-M99 是否支持深海环境运行？",
    "SolarWizard 逆变器如何接入 EnValt？",
    "EnValt 的火星基地储能系统如何配置？",
]

# TXT files are indexed as one parent document each, so they do not have rows.
TXT_CASES = {
    r"大文本切片\[APP side]Manual Information.txt":
        "How do I download and install the EnValt App?",
    r"大文本切片\[WEB side]Manual Information.txt":
        "What are the user account types in the EnValt iEnOS Cloud platform?",
    r"大文本切片\逆变器控制手册.txt":
        "What basic settings are available for the FX-3K/3.68K/4K/5K/6K-SL inverter?",
}


def field_text(value: Any) -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, default=str).strip()


def normalize_query(value: str) -> str:
    return " ".join(value.lower().split())


def normalized_column(value: Any) -> str:
    return str(value).strip().lower()


def is_atomic_query(query: str) -> bool:
    if len(query) < 8 or len(query) > 400:
        return False
    numbered_questions = re.findall(r"(?m)^\s*\d+\s*[.)、]\s*", query)
    if len(numbered_questions) >= 4:
        return False
    question_marks = query.count("?") + query.count("？")
    return question_marks <= 3


def discover_excluded_queries() -> set[str]:
    excluded: set[str] = set()
    paths = list(EVALUATION_DIR.glob("retrieval_cases*.json"))
    paths += list((EVALUATION_DIR / "holdout").glob("*.json"))
    for path in paths:
        if path.name in {DEFAULT_OUTPUT.name, DEFAULT_MANIFEST.name}:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, list):
            continue
        for case in payload:
            if not isinstance(case, dict):
                continue
            query = field_text(case.get("query"))
            if query:
                excluded.add(normalize_query(query))
    return excluded


def select_qa_columns(dataframe: pd.DataFrame) -> tuple[Any, Any]:
    columns = list(dataframe.columns)
    if len(columns) < 2:
        raise ValueError("表格至少需要两列")
    question = next(
        (column for column in columns if normalized_column(column) in QUESTION_COLUMNS),
        columns[0],
    )
    answer = next(
        (column for column in columns if normalized_column(column) in ANSWER_COLUMNS),
        columns[1],
    )
    if question == answer:
        raise ValueError("无法区分问题列和答案列")
    return question, answer


def row_content(row: pd.Series) -> str:
    # Keep this identical to rag.knowledge_retriever._load_knowledge_corpus.
    fields = [
        f"{column}: {value}"
        for column, value in row.items()
        if pd.notna(value)
    ]
    return "\n".join(fields).strip()


def load_excel_candidates(
    root: Path,
    excluded_queries: set[str],
) -> tuple[dict[str, deque[dict[str, Any]]], Counter[str]]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped: Counter[str] = Counter()
    seen_queries = set(excluded_queries)
    seen_content: set[str] = set()

    for path in sorted(root.rglob("*.xlsx")):
        if path.name.startswith("~$"):
            skipped["temporary_excel_file"] += 1
            continue
        source = str(path.relative_to(root))
        dataframe = pd.read_excel(path, engine="openpyxl")
        question_column, answer_column = select_qa_columns(dataframe)

        for row_index, row in dataframe.iterrows():
            content = row_content(row)
            if len(content) < 10:
                skipped["empty_or_too_short_record"] += 1
                continue
            if content in seen_content:
                skipped["duplicate_indexed_content"] += 1
                continue
            seen_content.add(content)

            query = field_text(row.get(question_column))
            answer = field_text(row.get(answer_column))
            normalized = normalize_query(query)
            if not answer:
                skipped["empty_answer"] += 1
                continue
            if not is_atomic_query(query):
                skipped["non_atomic_query"] += 1
                continue
            if normalized in seen_queries:
                skipped["duplicate_or_previously_evaluated_query"] += 1
                continue
            seen_queries.add(normalized)
            candidates[source].append(
                {
                    "query": query,
                    "expected_source": source,
                    "expected_row": int(row_index) + 2,
                    "source_kind": "xlsx",
                }
            )

    output: dict[str, deque[dict[str, Any]]] = {}
    for source, rows in candidates.items():
        rows.sort(
            key=lambda case: hashlib.sha256(
                f"{source}:{case['expected_row']}:enbot-dev-v4".encode("utf-8")
            ).hexdigest()
        )
        output[source] = deque(rows)
    return output, skipped


def balanced_sample(
    candidates: dict[str, deque[dict[str, Any]]], count: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    active_sources = deque(sorted(candidates))
    while active_sources and len(selected) < count:
        source = active_sources.popleft()
        rows = candidates[source]
        if rows:
            selected.append(rows.popleft())
        if rows:
            active_sources.append(source)
    return selected


def corpus_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("~$"):
            continue
        if path.suffix.lower() not in {".xlsx", ".txt", ".json"}:
            continue
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positives", type=int, default=60)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--knowledge-root", type=Path)
    args = parser.parse_args()

    if args.positives < len(TXT_CASES) + 1:
        raise ValueError(f"正例数量至少为 {len(TXT_CASES) + 1}")

    load_dotenv(PROJECT_ROOT / ".env")
    root = args.knowledge_root or Path(
        os.getenv("KNOWLEDGE_FILE", PROJECT_ROOT / "knowledge_data")
    )
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"知识库目录不存在: {root}")

    excluded_queries = discover_excluded_queries()
    excel_candidates, skipped = load_excel_candidates(root, excluded_queries)
    required_excel = args.positives - len(TXT_CASES)
    positives = balanced_sample(excel_candidates, required_excel)
    if len(positives) != required_excel:
        raise RuntimeError(
            f"只找到 {len(positives)} 条 Excel 正例，要求 {required_excel} 条"
        )

    available_sources = {str(path.relative_to(root)) for path in root.rglob("*.txt")}
    missing_txt = sorted(set(TXT_CASES) - available_sources)
    if missing_txt:
        raise FileNotFoundError(f"缺少预期长文本来源: {missing_txt}")
    positives.extend(
        {
            "query": query,
            "expected_source": source,
            "source_kind": "txt",
        }
        for source, query in TXT_CASES.items()
    )

    for index, case in enumerate(positives, start=1):
        case["name"] = f"DEV4-POS-{index:03d}"
    negatives = [
        {
            "name": f"DEV4-NEG-{index:03d}",
            "query": query,
            "expect_empty": True,
        }
        for index, query in enumerate(NEGATIVE_QUERIES, start=1)
    ]
    cases = positives + negatives

    normalized_queries = [normalize_query(case["query"]) for case in cases]
    if len(normalized_queries) != len(set(normalized_queries)):
        raise RuntimeError("V4 中存在重复问题")
    if any(query in excluded_queries for query in normalized_queries):
        raise RuntimeError("V4 中存在与旧评测集相同的问题")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    source_counts = Counter(case["expected_source"] for case in positives)
    source_files = sorted(source_counts)
    manifest = {
        "schema_version": 4,
        "purpose": "V2 XLSX/TXT knowledge-base retrieval development set",
        "knowledge_root": str(root),
        "corpus_sha256": corpus_fingerprint(root),
        "case_file": str(args.output.resolve()),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "total_count": len(cases),
        "covered_source_count": len(source_files),
        "covered_sources": source_counts,
        "excluded_old_query_count": len(excluded_queries),
        "skipped_candidate_counts": dict(sorted(skipped.items())),
        "label_rules": {
            "xlsx_row": "pandas row index + 2, matching production ingestion",
            "txt_row": "no row; the whole TXT file is one parent document",
            "path_separator": "native source path; evaluator normalizes slash direction",
            "positive_match": "exact source and row (or source only for TXT)",
            "negative_match": "retriever returns zero documents",
        },
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"知识库: {root}")
    print(f"输出文件: {args.output.resolve()}")
    print(f"清单文件: {args.manifest.resolve()}")
    print(f"正例/负例/总数: {len(positives)}/{len(negatives)}/{len(cases)}")
    print(f"覆盖来源: {len(source_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
