# EnBot：本地储能智能客服

EnBot 是一个本地运行的储能领域智能客服工程，组合了 Milvus 混合检索、
CPU 重排、拒答保护、本地微调模型、FastAPI、飞书机器人和 OCR/VLM 多模态诊断。

> 本公开仓库只包含技术代码、配置模板、测试和部署示例。它不包含任何公司知识库、
> 训练/验证/测试数据、逐样本评测结果、模型权重、数据库或真实凭据。

## 核心能力

- Milvus 向量召回与 BM25 词法召回，通过加权 RRF 融合。
- 本地 BGE reranker 精排，并对域外问题、未知型号和低置信结果进行拒答。
- 子块召回后恢复父级问答，减少长答案被截断或遗漏。
- 通过 Ollama 调用本地量化模型，不依赖付费模型 API。
- PaddleOCR 提取精确文字、数值和单位，视觉模型提取结构信息，再进行证据融合。
- FastAPI 提供健康检查、对话、反馈收集和图片诊断接口。
- 飞书官方 SDK 长连接机器人集成。
- 检索、回答质量、性能和多模态模块的自动化测试代码。

## 架构

```text
文本问题
  ├─ Milvus dense recall
  └─ BM25 lexical recall
          ↓
  weighted RRF → CPU reranker → refusal guard → parent restoration
          ↓
       RAG prompt → local Ollama model → FastAPI / Feishu

图片
  → validation/preprocessing
  → PaddleOCR + local vision model
  → evidence fusion
  → the same RAG pipeline
```

模型微调与 Optuna 搜索代码位于独立仓库：
[corner-01/enbot-finetune](https://github.com/corner-01/enbot-finetune)。

## 快速开始

### 1. 安装

```powershell
git clone https://github.com/corner-01/enbot.git
Set-Location enbot
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

根据 `.env.example` 配置你自己的本地模型、Milvus、Redis 和知识库路径。知识库必须是
你有权使用的数据，且不要提交到 Git。

### 2. 启动依赖并建立索引

```powershell
.\scripts\09_start_milvus.ps1
ollama list
.\venv\Scripts\python.exe -X utf8 scripts\rebuild_milvus_index.py --check
.\venv\Scripts\python.exe -X utf8 scripts\rebuild_milvus_index.py --yes
```

### 3. 启动文本 API

```powershell
.\venv\Scripts\python.exe -m uvicorn app.service_api:app `
  --host 127.0.0.1 --port 8000 --workers 1
```

- Swagger：<http://127.0.0.1:8000/docs>
- 存活检查：<http://127.0.0.1:8000/health/live>
- 就绪检查：<http://127.0.0.1:8000/health/ready>

### 4. 可选：启动图片诊断 API

```powershell
py -3.10 -m venv services\vision\.venv
services\vision\.venv\Scripts\python.exe -m pip install `
  -r services\vision\requirements.txt
.\scripts\11_start_vision_api.ps1
```

视觉 API Swagger：<http://127.0.0.1:8011/docs>。

## 测试

```powershell
.\venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

视觉模块使用独立环境：

```powershell
.\services\vision\.venv\Scripts\python.exe -X utf8 `
  -m unittest discover -s tests -p "test_vision*.py" -v
```

## 目录

```text
app/                 FastAPI、对话与反馈存储
rag/                 Milvus、混合检索、重排和 RAG 编排
services/vision/     OCR、视觉模型、证据融合和图片 API
integrations/        飞书机器人集成
evaluation/          通用评测程序，不含任何评测数据或结果
deployment/          本地服务与 Ollama 配置模板
scripts/             启动、索引重建与维护脚本
tests/               单元与回归测试
docs/                API 与公开发布边界说明
```

## 数据与安全边界

- 只复制 `.env.example`，不要提交 `.env`。
- 不提交知识库、训练集、验证集、测试集、holdout、人工复核表或逐样本生成结果。
- 不提交模型、LoRA adapter、GGUF、Milvus/Redis 数据、日志、缓存或虚拟环境。
- 对外部署时请使用 HTTPS、API Key、访问控制、限流和防火墙。
- 发布前执行 [公开发布检查](docs/PUBLIC_RELEASE.md)。

## License

当前仓库暂未附加开源许可证。在代码所有权和公开授权确认前，请勿假定可用于商业分发。
