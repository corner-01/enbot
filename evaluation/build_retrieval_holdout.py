"""Build a locked retrieval holdout from manually curated paraphrases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEV_SET = Path(__file__).with_name("retrieval_cases_v2.json")
DEFAULT_OUTPUT = Path(__file__).parent / "holdout" / "retrieval_holdout_v1.json"


CURATED_POSITIVES: list[tuple[str, int, str]] = [
    ("120.json", 40, "On a sunny day, what charging and discharging behavior should I expect from Weather Optimization?"),
    ("CESS-HV-9.4K.json", 25, "For a CESS-HV-9.4K battery, what are black power cable 1, red power cable 2, and the black CAN communication cables used for?"),
    ("CESS-HY-25~50K_user-manual.json", 94, "For the CESS-HY-25~50K inverter, explain first-time startup, what to do if it malfunctions, how SOC calibration is handled, and the inverter's main function."),
    ("HC-EV-AC-071122K.json", 9, "Where can I find the on-site installation steps and switch/cable recommendations for both HC-EV-AC-7/11/22K-S0 and HC-EV-AC-7/11/22K-S1 chargers?"),
    ("HESS-Gateway-M1.json", 5, "Which tools should be prepared before installing a HESS-Gateway-M1?"),
    ("HESS-HY-S-12K-S_user-manual.json", 51, "On the HESS-HY-S-6K-S display home page, what is the center icon, how do I open historical data, and where are the advanced settings?"),
    ("HESS-HY-S-3~6K.json", 22, "How can I use the GRID indicator on a HESS-HY-S-3~6K to confirm grid connection, and what does a steady white light mean?"),
    ("HESS-HY-S-6K-S_user-manual.json", 22, "How should the positive and negative conductors from two PV strings be wired to a HESS-HY-S-6K-S inverter?"),
    ("HESS-HY-T-15~20K-M4_user-manual.json", 91, "For a HESS-HV-T-15/20K M4, what is the first-start procedure, what should I do if it fails to operate, and how is SOC calibration performed during use?"),
    ("HESS-HY-T-5~12K.json", 59, "Describe the HESS-HY-T-5~12K electrical connection area and its main components, including the DC-switch requirement for Australia."),
    ("HOME-ESS-HV-2.7K.json", 35, "What is the correct grounding procedure for a HOME-ESS-HV-2.7K battery?"),
    ("HOME-ESS-HV-5.12K.json", 7, "Is there a diagram explaining the safety labels used on a HOME-ESS-HV-5.12K battery?"),
    ("HOME-ESS-HV-5.3K.json", 22, "Can you provide the HOME-ESS-HV-3.7K BDU interface diagram and explain its alphanumeric connection codes?"),
    ("HOME-ESS-LV-3.2K.json", 25, "Which accessories are included in the HOME-ESS-LV-3.2K battery package, and is there a complete accessories table?"),
    ("HOME-ESS-LV-3.68K (1).json", 29, "Show the grounding steps for a HOME-ESS-LV-3.68K battery and the corresponding wiring diagram."),
    ("HOME-ESS-LV-5.12K.json", 6, "What safety precautions, cabling rules, and electrical requirements apply to a HOME-ESS-LV-5.12K battery?"),
    ("HOME-ESS-LV-9.4K.json", 22, "What standard or specialized tools are needed to install a HOME-ESS-LV-9.4K, and where is the tool list?"),
    ("[APP side]Update instructions.json", 2, "How do I download and install the EnValt app on the app side?"),
    ("[WEB side]Manual Information.json", 2, "Where can I find the web-side operating instructions?"),
    ("hanchu_raw_dataset.json", 156, "In the settings interface, how do I navigate to Meter type for the HESS and CESS inverter families and the EV Charger?"),
    ("iESS-M1.json", 14, "What details should I provide when requesting troubleshooting help for an iESS-M1/M3/M5 smart meter?"),
    ("oss文件.json", 238, "Where can I obtain the Spanish certification file 230152RECO03-A-M4-CER_HESS-HY-T-4~20K-M(4) for the HESS-HY-T-4~20K-M three-phase inverter?"),
    ("低压电池告警切片.json", 71, "HOME-ESS-LV-3.2K、HOME-ESS-LV-9.4K、HOME-ESS-LV-5.12K或HOME-ESS-LV-3.68K出现“加热膜故障”时，可能原因和处理方法分别是什么？"),
    ("单相逆变器问答.json", 2, "What does Error Code 5 indicate on a Hanchu single-phase hybrid inverter?"),
    ("售后.json", 23, "低压电池的BAT-Logger采集器指示灯不亮通常表示什么？"),
    ("场景化问答(1).json", 171, "What can cause batteries in the same system to drift out of balance again?"),
    ("场景化问答.json", 1, "MPPT阴影扫描（Shadow Scan）是什么功能，应该在哪里开启？"),
    ("垂直领域(1).json", 3, "N-PE接地检测检查的是什么，出现异常时代表什么？"),
    ("垂直领域.json", 1, "SG Ready代表什么，它主要用于什么场景？"),
    ("市场与解决方案.json", 3, "Hanchu公司目前大约有多少名员工？"),
]


NEGATIVE_QUERIES = [
    "推荐两部适合周末看的科幻电影。",
    "第一次去北京旅游应该安排哪些景点？",
    "Python怎样按照字典的value给列表排序？",
    "普通感冒应该吃什么药？",
    "最近哪只股票值得买？",
    "帮我写一封向老师请假的邮件。",
    "把“今天天气很好”翻译成西班牙语。",
    "一元二次方程的求根公式是什么？",
    "篮球比赛中的三秒违例是什么意思？",
    "白色衣服染色后应该怎么清洗？",
    "XESS-HY-Z-888K支持哪些量子通信协议？",
    "HESS-HY-X-404K出现E.999故障应该如何处理？",
    "HOME-ESS-LV-777K的外形尺寸和重量是多少？",
    "CESS-HY-888K通过了哪些国际航空认证？",
    "虚构的SolarMagic App如何绑定量子电池？",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_sources(root: Path, dev_set: Path) -> None:
    dev_cases = load_json(dev_set)
    used = {
        (case.get("expected_source"), case.get("expected_row"))
        for case in dev_cases
        if case.get("expected_source") and case.get("expected_row") is not None
    }
    duplicates = []
    for source, row_number, query in CURATED_POSITIVES:
        if (source, row_number) in used:
            duplicates.append((source, row_number))
        source_path = root / source
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source: {source_path}")
        payload = load_json(source_path)
        rows = payload if isinstance(payload, list) else [payload]
        if row_number < 1 or row_number > len(rows):
            raise IndexError(f"Invalid row {row_number} for {source}")
        if not query.strip():
            raise ValueError(f"Empty query for {source}:{row_number}")
    if duplicates:
        raise RuntimeError(f"Holdout overlaps development set: {duplicates}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-set", type=Path, default=DEFAULT_DEV_SET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Locked holdout already exists: {args.output}. "
            "Do not overwrite it after evaluation."
        )

    load_dotenv(PROJECT_ROOT / ".env")
    root = Path(os.getenv("KNOWLEDGE_FILE", PROJECT_ROOT / "knowledge_data"))
    validate_sources(root, args.dev_set)

    positives = [
        {
            "name": f"HOLDOUT-POS-{index:03d}",
            "query": query,
            "expected_source": source,
            "expected_row": row_number,
        }
        for index, (source, row_number, query) in enumerate(CURATED_POSITIVES, start=1)
    ]
    negatives = [
        {
            "name": f"HOLDOUT-NEG-{index:03d}",
            "query": query,
            "expect_empty": True,
        }
        for index, query in enumerate(NEGATIVE_QUERIES, start=1)
    ]
    cases = positives + negatives
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "LOCKED_UNEVALUATED",
        "generator": "curated_manual_paraphrases",
        "model": None,
        "positive_cases": len(positives),
        "negative_cases": len(negatives),
        "excluded_development_set": str(args.dev_set.resolve()),
        "sha256": digest,
        "policy": "Evaluate once. Do not tune retrieval rules against this file.",
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Locked holdout: {args.output}")
    print(f"Manifest: {manifest_path}")
    print(f"SHA256: {digest}")
    print(f"Cases: {len(positives)} positive + {len(negatives)} negative")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
