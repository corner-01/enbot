#!/usr/bin/env python3
"""Prepare reproducible local QLoRA fine-tuning assets inside WSL.

The script preserves the cleaned v1 dataset and creates:
- deterministic, language-stratified train/validation/test splits;
- tokenizer-length and split manifests;
- LLaMA-Factory train, smoke-test, baseline, evaluation, chat, and export configs;
- launch scripts, a dependency lock, and a reusable preflight checker.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(
    os.environ.get("ENBOT_FINETUNE_ROOT", "/home/yft/projects/enbot-finetune")
).resolve()
SOURCE_DATA = ROOT / "data/processed/v1/hanchu_sft_all.json"
MODEL_DIR = ROOT / "models/Qwen2.5-7B-Instruct"
V2_DIR = ROOT / "data/processed/v2"
CONFIG_DIR = ROOT / "configs"
SCRIPT_DIR = ROOT / "scripts"
REPORT_DIR = ROOT / "data/reports"
OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"
EVAL_DIR = ROOT / "evaluations"
MERGED_DIR = ROOT / "merged-models"
LF_DIR = ROOT / "LLaMA-Factory"
CUTOFF_LEN = 1536
SEED = 42


def write_text(path: Path, content: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | 0o111)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def contains_cjk(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", value))


def percentile(sorted_values: list[int], ratio: float) -> int:
    if not sorted_values:
        return 0
    index = math.ceil(ratio * len(sorted_values)) - 1
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def token_length(tokenizer: Any, row: dict[str, str]) -> int:
    user_content = row["instruction"]
    if row.get("input", "").strip():
        user_content += "\n\n" + row["input"].strip()
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": row["output"]},
    ]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    token_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return len(token_ids)


def split_bucket(rows: list[tuple[dict[str, str], int]], label: str) -> dict[str, list[tuple[dict[str, str], int]]]:
    ranked: list[tuple[str, dict[str, str], int]] = []
    for row, length in rows:
        split_key = normalize_text(row["instruction"]) + "\n" + normalize_text(row.get("input", ""))
        digest = hashlib.sha256(f"{SEED}:{label}:{split_key}".encode("utf-8")).hexdigest()
        ranked.append((digest, row, length))
    ranked.sort(key=lambda item: item[0])

    holdout_count = int(len(ranked) * 0.05 + 0.5)
    test = [(row, length) for _, row, length in ranked[:holdout_count]]
    validation = [(row, length) for _, row, length in ranked[holdout_count : holdout_count * 2]]
    train = [(row, length) for _, row, length in ranked[holdout_count * 2 :]]
    return {"train": train, "validation": validation, "test": test}


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


def base_model_block() -> str:
    return f"""### model
model_name_or_path: {MODEL_DIR}
quantization_bit: 4
quantization_method: bnb
quantization_type: nf4
double_quantization: true
trust_remote_code: true"""


def common_sft_block() -> str:
    return f"""{base_model_block()}
upcast_layernorm: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all

### dataset
dataset_dir: {V2_DIR}
template: qwen
cutoff_len: {CUTOFF_LEN}
train_on_prompt: false
packing: false
overwrite_cache: false
preprocessing_num_workers: 4
dataloader_num_workers: 2"""


def build_configs() -> None:
    train_config = f"""{common_sft_block()}
dataset: hanchu_sft_train_v2
eval_dataset: hanchu_sft_validation_v2

### output
output_dir: {OUTPUT_DIR / "qwen2.5-7b-hanchu-qlora"}
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
learning_rate: 1.0e-4
weight_decay: 0.01
max_grad_norm: 1.0
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_steps: 20
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

    smoke_config = f"""{common_sft_block()}
dataset: hanchu_sft_smoke_v2

### output
output_dir: {OUTPUT_DIR / "smoke-test"}
logging_steps: 1
save_strategy: "no"
plot_loss: true
overwrite_output_dir: true
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
gradient_checkpointing: true
optim: adamw_torch
learning_rate: 1.0e-4
max_steps: 1
bf16: true
tf32: true
seed: {SEED}
ddp_timeout: 180000000
"""

    baseline_config = f"""{base_model_block()}

### method
stage: sft
do_predict: true
finetuning_type: lora

### dataset
dataset_dir: {V2_DIR}
eval_dataset: hanchu_sft_test_v2
template: qwen
cutoff_len: {CUTOFF_LEN}
overwrite_cache: false
preprocessing_num_workers: 4
dataloader_num_workers: 2

### output
output_dir: {EVAL_DIR / "baseline-qwen2.5-7b"}
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

    finetuned_config = baseline_config.replace(
        "### method",
        f"adapter_name_or_path: {OUTPUT_DIR / 'qwen2.5-7b-hanchu-qlora'}\n\n### method",
        1,
    ).replace(
        str(EVAL_DIR / "baseline-qwen2.5-7b"),
        str(EVAL_DIR / "finetuned-qwen2.5-7b"),
        1,
    )

    chat_config = f"""model_name_or_path: {MODEL_DIR}
adapter_name_or_path: {OUTPUT_DIR / "qwen2.5-7b-hanchu-qlora"}
quantization_bit: 4
quantization_method: bnb
quantization_type: nf4
double_quantization: true
template: qwen
infer_backend: huggingface
trust_remote_code: true
"""

    export_config = f"""### Important: LoRA merging must load the original model without 4-bit quantization.
model_name_or_path: {MODEL_DIR}
adapter_name_or_path: {OUTPUT_DIR / "qwen2.5-7b-hanchu-qlora"}
template: qwen
trust_remote_code: true

### export
export_dir: {MERGED_DIR / "qwen2.5-7b-hanchu"}
export_size: 5
export_device: cpu
export_legacy_format: false
"""

    write_text(CONFIG_DIR / "qwen2.5_7b_qlora_sft.yaml", train_config)
    write_text(CONFIG_DIR / "qwen2.5_7b_qlora_smoke.yaml", smoke_config)
    write_text(CONFIG_DIR / "qwen2.5_7b_base_predict.yaml", baseline_config)
    write_text(CONFIG_DIR / "qwen2.5_7b_qlora_predict.yaml", finetuned_config)
    write_text(CONFIG_DIR / "qwen2.5_7b_qlora_chat.yaml", chat_config)
    write_text(CONFIG_DIR / "qwen2.5_7b_qlora_export.yaml", export_config)


def build_preflight_script() -> None:
    content = r'''#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path("__ROOT__")
MODEL_DIR = ROOT / "models/Qwen2.5-7B-Instruct"
DATA_DIR = ROOT / "data/processed/v2"
CUTOFF_LEN = 1536


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def load(name: str):
    path = DATA_DIR / name
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, list) and data, f"{name} is empty or invalid"
    for index, row in enumerate(data):
        assert set(row) == {"instruction", "input", "output"}, f"{name}[{index}] has unexpected fields"
        assert row["instruction"].strip(), f"{name}[{index}] has empty instruction"
        assert row["output"].strip(), f"{name}[{index}] has empty output"
    return data


required_model_files = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
]
for file_name in required_model_files:
    assert (MODEL_DIR / file_name).is_file(), f"Missing model file: {file_name}"

train = load("hanchu_sft_train.json")
validation = load("hanchu_sft_validation.json")
test = load("hanchu_sft_test.json")
assert (len(train), len(validation), len(test)) == (3266, 182, 182)

key_sets = []
for rows in (train, validation, test):
    keys = {normalize(row["instruction"]) + "\n" + normalize(row["input"]) for row in rows}
    assert len(keys) == len(rows), "Duplicate prompt found inside a split"
    key_sets.append(keys)
assert not (key_sets[0] & key_sets[1])
assert not (key_sets[0] & key_sets[2])
assert not (key_sets[1] & key_sets[2])

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
max_length = 0
for row in train + validation + test:
    user = row["instruction"] + (("\n\n" + row["input"]) if row["input"].strip() else "")
    messages = [{"role": "user", "content": user}, {"role": "assistant", "content": row["output"]}]
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    max_length = max(max_length, len(ids))
assert max_length <= CUTOFF_LEN, f"Token length {max_length} exceeds cutoff {CUTOFF_LEN}"

import torch
import bitsandbytes as bnb
import transformers
import datasets
import accelerate
import peft
import trl

assert torch.cuda.is_available(), "CUDA is unavailable"
free_bytes, total_bytes = torch.cuda.mem_get_info()
result = {
    "status": "PASS",
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
    "accelerate": accelerate.__version__,
    "peft": peft.__version__,
    "trl": trl.__version__,
    "bitsandbytes": bnb.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "gpu_total_gib": round(total_bytes / 1024**3, 2),
    "gpu_free_gib": round(free_bytes / 1024**3, 2),
    "dataset_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
    "max_tokens": max_length,
    "cutoff_len": CUTOFF_LEN,
    "split_overlap": 0,
}
print(json.dumps(result, ensure_ascii=False, indent=2))
if free_bytes < 14 * 1024**3:
    print("WARNING: Less than 14 GiB GPU memory is free. Stop Ollama/RAG model processes before training.")
''' .replace("__ROOT__", str(ROOT))
    write_text(SCRIPT_DIR / "preflight.py", content, executable=True)


def shell_prelude() -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source {ROOT / ".venv-llamafactory/bin/activate"}
cd {LF_DIR}
"""


def build_launch_scripts() -> None:
    scripts = {
        "00_preflight.sh": f"python {SCRIPT_DIR / 'preflight.py'}",
        "01_smoke_test.sh": f"llamafactory-cli train {CONFIG_DIR / 'qwen2.5_7b_qlora_smoke.yaml'}",
        "02_baseline_evaluate.sh": f"llamafactory-cli train {CONFIG_DIR / 'qwen2.5_7b_base_predict.yaml'}",
        "03_train.sh": (
            f"export TENSORBOARD_LOGGING_DIR={LOG_DIR / 'qwen2.5-7b-hanchu-qlora'}\n"
            f"llamafactory-cli train {CONFIG_DIR / 'qwen2.5_7b_qlora_sft.yaml'}"
        ),
        "04_finetuned_evaluate.sh": f"llamafactory-cli train {CONFIG_DIR / 'qwen2.5_7b_qlora_predict.yaml'}",
        "05_tensorboard.sh": f"tensorboard --logdir {LOG_DIR} --host 0.0.0.0 --port 6006",
        "06_chat.sh": f"llamafactory-cli chat {CONFIG_DIR / 'qwen2.5_7b_qlora_chat.yaml'}",
        "07_export.sh": f"llamafactory-cli export {CONFIG_DIR / 'qwen2.5_7b_qlora_export.yaml'}",
    }
    for name, command in scripts.items():
        write_text(SCRIPT_DIR / name, shell_prelude() + command, executable=True)


def build_readme(manifest: dict[str, Any]) -> None:
    readme = f"""# enbot 本地微调运行手册

## 已准备内容

- 原始模型：Qwen2.5-7B-Instruct，保存在本地，不调用任何付费 API。
- 方法：4-bit NF4 QLoRA，LoRA rank 16 / alpha 32 / dropout 0.05。
- 数据：{manifest["counts"]["train"]} 条训练、{manifest["counts"]["validation"]} 条验证、{manifest["counts"]["test"]} 条最终测试。
- 上下文长度：{CUTOFF_LEN} tokens；当前最长样本 {manifest["token_length"]["max"]} tokens，不会被截断。
- 有效批次：1 x 16 梯度累积 = 16；2 个 epoch，预计约 {manifest["estimated_optimizer_steps"]} 个优化步骤。
- 监控：本地 TensorBoard；不连接 WandB、OpenAI 或其他在线训练服务。

## 正式训练前按顺序执行

1. 在 VS Code 中打开 WSL 项目目录：{ROOT}
2. 停止正在占用 GPU 的 Ollama/RAG 模型。Redis 可以继续运行。
3. 运行完整检查：

~~~bash
{SCRIPT_DIR / "00_preflight.sh"}
~~~

4. 运行单步冒烟测试，确认模型加载、4-bit 量化、数据预处理和反向传播全部正常：

~~~bash
{SCRIPT_DIR / "01_smoke_test.sh"}
~~~

5. 建议先保存原模型在测试集上的基线结果（182 条，生成会比较慢）：

~~~bash
{SCRIPT_DIR / "02_baseline_evaluate.sh"}
~~~

6. 确认后才启动正式训练：

~~~bash
{SCRIPT_DIR / "03_train.sh"}
~~~

## 训练中查看

另开一个 WSL 终端运行：

~~~bash
{SCRIPT_DIR / "05_tensorboard.sh"}
~~~

Windows 浏览器访问 http://localhost:6006 。重点看 train/loss、eval/loss 和 learning_rate。

## 训练完成后

~~~bash
{SCRIPT_DIR / "04_finetuned_evaluate.sh"}
{SCRIPT_DIR / "06_chat.sh"}
~~~

测试集只用于最终对比，不要再把其中问题加入训练集。基线和微调后结果分别位于：

- {EVAL_DIR / "baseline-qwen2.5-7b"}
- {EVAL_DIR / "finetuned-qwen2.5-7b"}

确认效果后，可在内存充足时将 LoRA 合并到原模型：

~~~bash
{SCRIPT_DIR / "07_export.sh"}
~~~

## 断点续训

如果训练中断，不要删除输出目录。在 {CONFIG_DIR / "qwen2.5_7b_qlora_sft.yaml"} 中把
resume_from_checkpoint 改成最近的 checkpoint 绝对路径，再运行 03_train.sh。

## 关键文件

- 正式训练配置：{CONFIG_DIR / "qwen2.5_7b_qlora_sft.yaml"}
- 数据注册：{V2_DIR / "dataset_info.json"}
- 数据与拆分说明：{V2_DIR / "split_manifest.json"}
- Token 统计：{V2_DIR / "token_length_report.json"}
- 环境版本锁：{ROOT / "requirements-llamafactory-lock.txt"}
"""
    write_text(ROOT / "FINETUNE_README.md", readme)


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None


def main() -> None:
    if not SOURCE_DATA.is_file():
        raise FileNotFoundError(f"Cleaned source dataset not found: {SOURCE_DATA}")
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Local model not found: {MODEL_DIR}")

    for directory in (V2_DIR, CONFIG_DIR, SCRIPT_DIR, REPORT_DIR, OUTPUT_DIR, LOG_DIR, EVAL_DIR, MERGED_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    source_rows = json.loads(SOURCE_DATA.read_text(encoding="utf-8"))
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("The v1 source dataset is empty or not a JSON array")

    clean_rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, original in enumerate(source_rows):
        row = {
            "instruction": str(original.get("instruction", "")).strip(),
            "input": str(original.get("input", "")).strip(),
            "output": str(original.get("output", "")).strip(),
        }
        if not row["instruction"] or not row["output"]:
            raise ValueError(f"Empty instruction/output at source index {index}")
        key = normalize_text(row["instruction"]) + "\n" + normalize_text(row["input"])
        if key in seen:
            raise ValueError(f"Duplicate normalized prompt at source index {index}")
        seen.add(key)
        clean_rows.append(row)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
    rows_with_lengths = [(row, token_length(tokenizer, row)) for row in clean_rows]
    all_lengths = sorted(length for _, length in rows_with_lengths)
    if all_lengths[-1] > CUTOFF_LEN:
        raise ValueError(f"Longest sample is {all_lengths[-1]} tokens, greater than cutoff {CUTOFF_LEN}")

    buckets = {
        "cjk": [(row, length) for row, length in rows_with_lengths if contains_cjk(row["instruction"] + row["output"])],
        "non_cjk": [
            (row, length) for row, length in rows_with_lengths if not contains_cjk(row["instruction"] + row["output"])
        ],
    }
    combined = {"train": [], "validation": [], "test": []}
    bucket_counts: dict[str, dict[str, int]] = {}
    for label, rows in buckets.items():
        result = split_bucket(rows, label)
        bucket_counts[label] = {name: len(values) for name, values in result.items()}
        for name in combined:
            combined[name].extend(result[name])

    for name in combined:
        combined[name].sort(
            key=lambda item: hashlib.sha256(
                (normalize_text(item[0]["instruction"]) + "\n" + normalize_text(item[0]["input"])).encode("utf-8")
            ).hexdigest()
        )

    paths = {
        "train": V2_DIR / "hanchu_sft_train.json",
        "validation": V2_DIR / "hanchu_sft_validation.json",
        "test": V2_DIR / "hanchu_sft_test.json",
    }
    for name, path in paths.items():
        write_json(path, [row for row, _ in combined[name]])

    longest_row, longest_length = max(rows_with_lengths, key=lambda item: item[1])
    smoke_path = V2_DIR / "hanchu_sft_smoke.json"
    write_json(smoke_path, [longest_row])

    dataset_info = {
        "hanchu_sft_train_v2": dataset_registration(paths["train"].name),
        "hanchu_sft_validation_v2": dataset_registration(paths["validation"].name),
        "hanchu_sft_test_v2": dataset_registration(paths["test"].name),
        "hanchu_sft_smoke_v2": dataset_registration(smoke_path.name),
    }
    write_json(V2_DIR / "dataset_info.json", dataset_info)

    split_keys: dict[str, set[str]] = {}
    for name in combined:
        split_keys[name] = {
            normalize_text(row["instruction"]) + "\n" + normalize_text(row["input"]) for row, _ in combined[name]
        }
    overlaps = {
        "train_validation": len(split_keys["train"] & split_keys["validation"]),
        "train_test": len(split_keys["train"] & split_keys["test"]),
        "validation_test": len(split_keys["validation"] & split_keys["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Dataset split leakage detected: {overlaps}")

    train_count = len(combined["train"])
    estimated_steps = math.ceil(train_count / 16) * 2
    manifest = {
        "version": "v2",
        "source": str(SOURCE_DATA),
        "source_sha256": sha256_file(SOURCE_DATA),
        "seed": SEED,
        "split_strategy": "SHA-256 deterministic split, stratified by CJK/non-CJK; 90%/5%/5%",
        "counts": {name: len(values) for name, values in combined.items()},
        "language_bucket_counts": bucket_counts,
        "normalized_prompt_overlap": overlaps,
        "files": {
            name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()
        },
        "smoke_sample": {"path": str(smoke_path), "tokens": longest_length},
        "token_length": {
            "p50": percentile(all_lengths, 0.50),
            "p90": percentile(all_lengths, 0.90),
            "p95": percentile(all_lengths, 0.95),
            "p99": percentile(all_lengths, 0.99),
            "max": all_lengths[-1],
            "over_512": sum(length > 512 for length in all_lengths),
            "over_1024": sum(length > 1024 for length in all_lengths),
            "over_1536": sum(length > 1536 for length in all_lengths),
        },
        "training": {
            "method": "QLoRA",
            "quantization": "bitsandbytes 4-bit NF4 with double quantization",
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "cutoff_len": CUTOFF_LEN,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "effective_batch_size": 16,
            "epochs": 2,
            "learning_rate": 0.0001,
        },
        "estimated_optimizer_steps": estimated_steps,
    }
    write_json(V2_DIR / "split_manifest.json", manifest)
    write_json(
        V2_DIR / "token_length_report.json",
        {
            "tokenizer": str(MODEL_DIR),
            "chat_template": "Qwen2.5 tokenizer chat template",
            "sample_count": len(all_lengths),
            "cutoff_len": CUTOFF_LEN,
            **manifest["token_length"],
        },
    )

    build_configs()
    build_preflight_script()
    build_launch_scripts()
    build_readme(manifest)

    freeze = command_output([str(ROOT / ".venv-llamafactory/bin/python"), "-m", "pip", "freeze"])
    if freeze:
        write_text(ROOT / "requirements-llamafactory-lock.txt", freeze)

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "llamafactory_commit": command_output(["git", "-C", str(LF_DIR), "rev-parse", "HEAD"]),
        "gpu": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "model_path": str(MODEL_DIR),
        "model_config_sha256": sha256_file(MODEL_DIR / "config.json"),
        "dataset_manifest": str(V2_DIR / "split_manifest.json"),
        "network_api_required": False,
    }
    write_json(ROOT / "environment_manifest.json", environment)

    print(json.dumps({
        "status": "prepared",
        "root": str(ROOT),
        "counts": manifest["counts"],
        "max_tokens": manifest["token_length"]["max"],
        "cutoff_len": CUTOFF_LEN,
        "estimated_optimizer_steps": estimated_steps,
        "readme": str(ROOT / "FINETUNE_README.md"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
