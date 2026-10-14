#!/usr/bin/env python3
"""Build the EnBot V9 SFT dataset and isolated QLoRA training assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


SEED = 42
CUTOFF_LEN = 1536
EXPECTED_COUNTS = {"train": 3705, "validation": 206, "test": 206}


def normalize(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def contains_cjk(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | 0o111)


def row_key(row: dict[str, str]) -> tuple[str, str]:
    return normalize(row["instruction"]), normalize(row.get("input", ""))


def exact_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (*row_key(row), normalize(row["output"]))


def token_length(tokenizer: Any, row: dict[str, str]) -> int:
    user = row["instruction"]
    if row.get("input", "").strip():
        user += "\n\n" + row["input"].strip()
    encoded = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["output"]},
        ],
        tokenize=True,
        add_generation_prompt=False,
    )
    token_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return len(token_ids)


def percentile(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * ratio) - 1))
    return ordered[index]


def split_bucket(items: list[dict[str, Any]], label: str) -> dict[str, list[dict[str, Any]]]:
    ranked = []
    for item in items:
        prompt = "\n".join(row_key(item["row"]))
        digest = hashlib.sha256(f"{SEED}:{label}:{prompt}".encode("utf-8")).hexdigest()
        ranked.append((digest, item))
    ranked.sort(key=lambda pair: pair[0])
    holdout_count = int(len(ranked) * 0.05 + 0.5)
    return {
        "test": [item for _, item in ranked[:holdout_count]],
        "validation": [item for _, item in ranked[holdout_count : holdout_count * 2]],
        "train": [item for _, item in ranked[holdout_count * 2 :]],
    }


def stable_split(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {"zh": [], "non_zh": []}
    for item in items:
        row = item["row"]
        text = row["instruction"] + "\n" + row.get("input", "")
        buckets["zh" if contains_cjk(text) else "non_zh"].append(item)

    output = {"train": [], "validation": [], "test": []}
    for label in ("non_zh", "zh"):
        split = split_bucket(buckets[label], label)
        for name in output:
            output[name].extend(split[name])

    for name, rows in output.items():
        rows.sort(
            key=lambda item: hashlib.sha256(
                f"{SEED}:{name}:{row_key(item['row'])}".encode("utf-8")
            ).hexdigest()
        )
    return output


def dataset_registration(file_name: str) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    }


def build_configs(root: Path) -> None:
    model = root / "models/Qwen2.5-7B-Instruct"
    data = root / "data/processed/v9"
    adapter = root / "outputs/qwen2.5-7b-envalt-v9-qlora"
    config_dir = root / "configs"
    eval_root = root / "evaluations"

    model_block = f"""### model
model_name_or_path: {model}
quantization_bit: 4
quantization_method: bnb
quantization_type: nf4
double_quantization: true
trust_remote_code: true
upcast_layernorm: true"""
    method_block = """### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.08
lora_target: all"""
    dataset_block = f"""### dataset
dataset_dir: {data}
dataset: envalt_sft_train_v9
eval_dataset: envalt_sft_validation_v9
template: qwen
cutoff_len: {CUTOFF_LEN}
train_on_prompt: false
packing: false
overwrite_cache: false
preprocessing_num_workers: 4
dataloader_num_workers: 2"""
    train_config = f"""{model_block}

{method_block}

{dataset_block}

### output
output_dir: {adapter}
logging_steps: 10
save_strategy: steps
save_steps: 100
save_total_limit: 3
plot_loss: true
overwrite_output_dir: false
save_only_model: false
report_to: tensorboard

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
per_device_eval_batch_size: 1
gradient_checkpointing: true
optim: adamw_torch
learning_rate: 0.000144499544513933
weight_decay: 0.03
max_grad_norm: 1.0
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
tf32: true
seed: {SEED}
data_seed: {SEED}
ddp_timeout: 180000000
resume_from_checkpoint: null

### validation
eval_strategy: steps
eval_steps: 100
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false
"""
    smoke_config = train_config.replace(
        "dataset: envalt_sft_train_v9\neval_dataset: envalt_sft_validation_v9",
        "dataset: envalt_sft_smoke_v9",
    ).replace(
        f"output_dir: {adapter}",
        f"output_dir: {root / 'outputs/v9-smoke-test'}",
    ).replace(
        "logging_steps: 10\nsave_strategy: steps\nsave_steps: 100\nsave_total_limit: 3",
        "logging_steps: 1\nsave_strategy: 'no'",
    ).replace(
        "overwrite_output_dir: false",
        "overwrite_output_dir: true",
    ).replace(
        "gradient_accumulation_steps: 16",
        "gradient_accumulation_steps: 1",
    ).replace(
        "num_train_epochs: 2.0",
        "max_steps: 1",
    ).replace(
        "report_to: tensorboard",
        "report_to: none",
    ).split("### validation", 1)[0]

    predict_base = f"""{model_block}

### method
stage: sft
do_predict: true
finetuning_type: lora

### dataset
dataset_dir: {data}
eval_dataset: envalt_sft_test_v9
template: qwen
cutoff_len: {CUTOFF_LEN}
overwrite_cache: false
preprocessing_num_workers: 4
dataloader_num_workers: 2

### output
output_dir: {eval_root / 'v9-baseline-qwen2.5-7b'}
overwrite_output_dir: false
report_to: none

### generation
per_device_eval_batch_size: 1
predict_with_generate: true
do_sample: false
max_new_tokens: 512
repetition_penalty: 1.05
ddp_timeout: 180000000
"""
    predict_tuned = predict_base.replace(
        "### method",
        f"adapter_name_or_path: {adapter}\n\n### method",
        1,
    ).replace(
        str(eval_root / "v9-baseline-qwen2.5-7b"),
        str(eval_root / "v9-finetuned-qwen2.5-7b"),
        1,
    )
    chat = f"""model_name_or_path: {model}
adapter_name_or_path: {adapter}
quantization_bit: 4
quantization_method: bnb
quantization_type: nf4
double_quantization: true
template: qwen
infer_backend: huggingface
trust_remote_code: true
"""
    export = f"""model_name_or_path: {model}
adapter_name_or_path: {adapter}
template: qwen
trust_remote_code: true

### export
export_dir: {root / 'merged-models/qwen2.5-7b-envalt-v9'}
export_size: 5
export_device: cpu
export_legacy_format: false
"""
    files = {
        "qwen2_5_7b_v9_qlora_sft.yaml": train_config,
        "qwen2_5_7b_v9_qlora_smoke.yaml": smoke_config,
        "qwen2_5_7b_v9_base_predict.yaml": predict_base,
        "qwen2_5_7b_v9_qlora_predict.yaml": predict_tuned,
        "qwen2_5_7b_v9_qlora_chat.yaml": chat,
        "qwen2_5_7b_v9_qlora_export.yaml": export,
    }
    for name, content in files.items():
        write_text(config_dir / name, content)


def build_launchers(root: Path) -> None:
    prefix = f"""#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source {root / '.venv-llamafactory/bin/activate'}
cd {root / 'LLaMA-Factory'}"""
    launchers = {
        "81_preflight_v9.sh": f"{prefix}\npython {root / 'scripts/81_preflight_v9.py'}",
        "82_smoke_v9.sh": f"{prefix}\nllamafactory-cli train {root / 'configs/qwen2_5_7b_v9_qlora_smoke.yaml'}",
        "83_train_v9.sh": f"{prefix}\nexport TENSORBOARD_LOGGING_DIR={root / 'logs/qwen2.5-7b-envalt-v9-qlora'}\nllamafactory-cli train {root / 'configs/qwen2_5_7b_v9_qlora_sft.yaml'}",
        "84_tensorboard_v9.sh": f"{prefix}\ntensorboard --logdir {root / 'outputs/qwen2.5-7b-envalt-v9-qlora'} --host 0.0.0.0 --port 6006",
        "85_baseline_evaluate_v9.sh": f"{prefix}\nllamafactory-cli train {root / 'configs/qwen2_5_7b_v9_base_predict.yaml'}",
        "86_finetuned_evaluate_v9.sh": f"{prefix}\nllamafactory-cli train {root / 'configs/qwen2_5_7b_v9_qlora_predict.yaml'}",
        "87_chat_v9.sh": f"{prefix}\nllamafactory-cli chat {root / 'configs/qwen2_5_7b_v9_qlora_chat.yaml'}",
        "88_export_v9.sh": f"{prefix}\nllamafactory-cli export {root / 'configs/qwen2_5_7b_v9_qlora_export.yaml'}",
    }
    for name, content in launchers.items():
        write_text(root / "scripts" / name, content, executable=True)


def build_preflight(root: Path) -> None:
    content = r'''#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("__ROOT__")
DATA = ROOT / "data/processed/v9"
MODEL = ROOT / "models/Qwen2.5-7B-Instruct"
EXPECTED = {
    "envalt_sft_train.json": 3705,
    "envalt_sft_validation.json": 206,
    "envalt_sft_test.json": 206,
}


def normalize(value: str) -> str:
    import unicodedata
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


manifest = json.loads((DATA / "split_manifest.json").read_text(encoding="utf-8"))
sets = {}
counts = {}
for file_name, expected in EXPECTED.items():
    rows = json.loads((DATA / file_name).read_text(encoding="utf-8"))
    assert len(rows) == expected, f"{file_name}: {len(rows)} != {expected}"
    keys = set()
    for index, row in enumerate(rows):
        assert set(row) == {"instruction", "input", "output"}, (file_name, index)
        assert row["instruction"].strip(), (file_name, index, "empty instruction")
        assert row["output"].strip(), (file_name, index, "empty output")
        key = normalize(row["instruction"]) + "\n" + normalize(row["input"])
        assert key not in keys, (file_name, index, "duplicate prompt")
        keys.add(key)
    split = file_name.removeprefix("envalt_sft_").removesuffix(".json")
    sets[split] = keys
    counts[split] = len(rows)
    assert sha256_file(DATA / file_name) == manifest["sha256"][split]

assert not sets["train"] & sets["validation"]
assert not sets["train"] & sets["test"]
assert not sets["validation"] & sets["test"]
assert sum(counts.values()) == 4117

token_report = json.loads((DATA / "token_length_report.json").read_text(encoding="utf-8"))
assert token_report["max"] <= token_report["cutoff_len"] == 1536
cleaning = json.loads((DATA / "cleaning_report.json").read_text(encoding="utf-8"))
assert cleaning["accepted_rows"] == 4117
assert cleaning["split_counts"] == counts

for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"):
    assert (MODEL / name).is_file(), f"Missing model file: {name}"

import accelerate
import bitsandbytes
import datasets
import peft
import torch
import transformers
import trl

assert torch.cuda.is_available(), "CUDA is unavailable"
free_bytes, total_bytes = torch.cuda.mem_get_info()
print(json.dumps({
    "status": "PASS",
    "counts": counts,
    "total": sum(counts.values()),
    "max_tokens": token_report["max"],
    "cutoff_len": token_report["cutoff_len"],
    "cuda_device": torch.cuda.get_device_name(0),
    "gpu_free_gib": round(free_bytes / 1024**3, 2),
    "gpu_total_gib": round(total_bytes / 1024**3, 2),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
}, ensure_ascii=False, indent=2))
'''.replace("__ROOT__", str(root))
    write_text(root / "scripts/81_preflight_v9.py", content, executable=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/yft/projects/enbot-finetune"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    source_dir = args.source_dir.expanduser().resolve()
    model_dir = root / "models/Qwen2.5-7B-Instruct"
    target = root / "data/processed/v9"
    staging = root / "data/processed/v9.building"
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing {target}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    files = sorted(source_dir.glob("*.json"))
    invalid: list[dict[str, Any]] = []
    exact: dict[tuple[str, str, str], dict[str, Any]] = {}
    valid_rows = 0
    duplicate_count = 0
    raw_rows = 0
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = payload if isinstance(payload, list) else [payload]
        raw_rows += len(rows)
        for source_row, value in enumerate(rows, start=1):
            if not isinstance(value, dict):
                invalid.append({"source": path.name, "row": source_row, "reason": "not_object"})
                continue
            row = {
                field: str(value.get(field, "")).strip()
                for field in ("instruction", "input", "output")
            }
            if not row["instruction"] or not row["output"]:
                invalid.append({"source": path.name, "row": source_row, "reason": "empty_instruction_or_output"})
                continue
            valid_rows += 1
            key = exact_key(row)
            if key in exact:
                duplicate_count += 1
                continue
            exact[key] = {"row": row, "source": path.name, "source_row": source_row}

    by_prompt: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in exact.values():
        by_prompt[row_key(item["row"])].append(item)
    conflicts = []
    safe = []
    for prompt_key, items in by_prompt.items():
        if len({normalize(item["row"]["output"]) for item in items}) > 1:
            conflicts.append(
                {
                    "instruction": items[0]["row"]["instruction"],
                    "input": items[0]["row"]["input"],
                    "candidates": [
                        {
                            "source": item["source"],
                            "source_row": item["source_row"],
                            "output_preview": item["row"]["output"][:500],
                        }
                        for item in items
                    ],
                }
            )
        else:
            safe.append(items[0])

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
    )
    accepted = []
    over_cutoff = []
    lengths = []
    for item in safe:
        length = token_length(tokenizer, item["row"])
        item["token_length"] = length
        if length > CUTOFF_LEN:
            over_cutoff.append(
                {
                    "source": item["source"],
                    "source_row": item["source_row"],
                    "instruction": item["row"]["instruction"],
                    "token_length": length,
                }
            )
            continue
        accepted.append(item)
        lengths.append(length)

    split = stable_split(accepted)
    counts = {name: len(items) for name, items in split.items()}
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Unexpected V9 split counts: {counts} != {EXPECTED_COUNTS}")

    split_keys = {
        name: {row_key(item["row"]) for item in items}
        for name, items in split.items()
    }
    assert not split_keys["train"] & split_keys["validation"]
    assert not split_keys["train"] & split_keys["test"]
    assert not split_keys["validation"] & split_keys["test"]

    output_names = {
        "train": "envalt_sft_train.json",
        "validation": "envalt_sft_validation.json",
        "test": "envalt_sft_test.json",
    }
    for name, file_name in output_names.items():
        write_json(staging / file_name, [item["row"] for item in split[name]])

    zh = [item for item in split["train"] if contains_cjk(item["row"]["instruction"] + item["row"]["input"])]
    non_zh = [item for item in split["train"] if item not in zh]
    smoke = [item["row"] for item in non_zh[:4] + zh[:4]]
    write_json(staging / "envalt_sft_smoke.json", smoke)
    write_json(
        staging / "dataset_info.json",
        {
            "envalt_sft_train_v9": dataset_registration("envalt_sft_train.json"),
            "envalt_sft_validation_v9": dataset_registration("envalt_sft_validation.json"),
            "envalt_sft_test_v9": dataset_registration("envalt_sft_test.json"),
            "envalt_sft_smoke_v9": dataset_registration("envalt_sft_smoke.json"),
        },
    )
    write_json(staging / "conflict_report.json", {"count": len(conflicts), "groups": conflicts})
    write_json(staging / "over_cutoff_report.json", {"count": len(over_cutoff), "rows": over_cutoff})
    write_json(staging / "invalid_report.json", {"count": len(invalid), "rows": invalid})
    write_json(
        staging / "token_length_report.json",
        {
            "cutoff_len": CUTOFF_LEN,
            "count": len(lengths),
            "min": min(lengths),
            "p50": percentile(lengths, 0.50),
            "p90": percentile(lengths, 0.90),
            "p95": percentile(lengths, 0.95),
            "p99": percentile(lengths, 0.99),
            "max": max(lengths),
            "excluded_over_cutoff": len(over_cutoff),
        },
    )
    provenance = []
    for name, items in split.items():
        for item in items:
            provenance.append(
                {
                    "split": name,
                    "fingerprint": hashlib.sha256(
                        "\n".join(exact_key(item["row"])).encode("utf-8")
                    ).hexdigest(),
                    "source": item["source"],
                    "source_row": item["source_row"],
                    "token_length": item["token_length"],
                }
            )
    write_text(
        staging / "source_map.jsonl",
        "\n".join(json.dumps(item, ensure_ascii=False) for item in provenance),
    )
    language_counts = Counter(
        "zh" if contains_cjk(item["row"]["instruction"] + item["row"]["input"]) else "non_zh"
        for item in accepted
    )
    cleaning_report = {
        "source_directory": str(source_dir),
        "source_files": len(files),
        "raw_rows": raw_rows,
        "valid_rows_before_deduplication": valid_rows,
        "invalid_rows": len(invalid),
        "normalized_exact_duplicates_removed": duplicate_count,
        "unique_rows_before_conflict_filter": len(exact),
        "conflict_prompt_groups_removed": len(conflicts),
        "conflict_rows_removed": sum(len(group["candidates"]) for group in conflicts),
        "over_cutoff_rows_removed": len(over_cutoff),
        "accepted_rows": len(accepted),
        "language_counts": dict(language_counts),
        "split_counts": counts,
        "seed": SEED,
        "cutoff_len": CUTOFF_LEN,
    }
    write_json(staging / "cleaning_report.json", cleaning_report)

    file_hashes = {
        name: sha256_file(staging / file_name)
        for name, file_name in output_names.items()
    }
    write_json(
        staging / "split_manifest.json",
        {
            "version": "v9",
            "strategy": "normalized-dedup, conflict-exclusion, language-stratified deterministic 90/5/5",
            "counts": counts,
            "sha256": file_hashes,
            "holdout_policy": "test split is locked; do not use it for tuning or data revision",
        },
    )

    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)
    build_configs(root)
    build_preflight(root)
    build_launchers(root)
    print(json.dumps({"status": "PASS", **cleaning_report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
