# EnBot：储能领域智能客服

EnBot 是一个完全本地运行的储能领域智能客服系统，组合了微调大模型、
Milvus 混合检索、CPU 重排、拒答机制、多模态 OCR/VLM 诊断、FastAPI 和
飞书机器人。运行过程不依赖付费模型 API。

> 公开仓库不包含公司知识库、训练/评测正文、模型权重或任何凭据。
> 使用者需要准备自己有权使用的数据和本地模型。

## 核心能力

- **混合 RAG**：Milvus 稠密向量 + BM25 词法召回，通过加权 RRF 融合。
- **精排与拒答**：使用本地 BGE reranker，并对未知型号、故障码和域外问题保守拒答。
- **长文本恢复**：子块负责召回，命中后恢复完整父级问答或目标章节。
- **微调模型**：通过 Ollama 运行基于新数据训练的 EnBot V9 模型，部署为 Q4_K_M。
- **多模态诊断**：PaddleOCR 提取精确文字、数值和单位，Qwen3-VL 提取图像结构，证据融合后再进入 RAG。
- **服务接口**：文本 API（8000）和视觉 API（8011）独立部署，支持健康检查、API Key、并发限制与会话隔离。
- **业务接入**：支持飞书官方 SDK 长连接机器人和用户反馈收集。
- **离线评测**：检索、拒答、回答质量 A/B、性能和多模态组件均有自动化测试。

## 系统架构

```text
文本问题
   │
   ├─ Milvus Dense ─┐
   ├─ BM25 ─────────┼─ Weighted RRF ─ CPU Reranker ─ 拒答/父级恢复 ─ RAG Prompt
   │                 │
   └─────────────────┘
                                                      │
图片 ─ 校验/预处理 ─┬─ PaddleOCR ─┐                  │
                    └─ Qwen3-VL ──┴─ 证据融合 ───────┘
                                                      │
                                    enbot-v9:7b / Ollama
                                                      │
                                      FastAPI / 飞书 / Postman
```

模型训练、Optuna 搜索和评测代码位于独立仓库：
[corner-01/enbot-finetune](https://github.com/corner-01/enbot-finetune)。

## 运行要求

已验证开发环境：Windows 11、WSL2、Python 3.10/3.12、Docker Desktop、
NVIDIA RTX 4070 Ti SUPER 16 GiB。

- 16 GiB GPU 可通过模型分时加载或部分 CPU offload 完成开发演示。
- 若希望 8B BF16 Embedding 与 7B Q4 生成模型同时完整驻留 GPU，建议 32 GiB 显存。
- 建议系统内存不少于 32 GiB，磁盘可用空间不少于 100 GiB。

## 快速开始

### 1. 克隆并安装文本服务

```powershell
git clone https://github.com/corner-01/enbot.git
Set-Location enbot
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### 2. 准备本地资产

以下内容不会上传 GitHub，请按 `.env` 中的路径自行放置：

```text
models/Qwen3-8B-native/
models/bge-reranker-v2-m3/
knowledge_data/
deployment/ollama-v9/enbot-v9-q4_k_m.gguf
```

知识库必须是你有权使用的资料。不要把示例中的公司内部数据替换回公开仓库。

### 3. 启动基础服务

1. 启动 Docker Desktop。
2. 启动 Milvus：

```powershell
.\scripts\09_start_milvus.ps1
```

3. 确认 Ollama 已运行，并导入本地 GGUF：

```powershell
ollama create enbot-v9:7b -f deployment\ollama-v9\Modelfile
ollama list
```

4. 首次构建 Milvus 索引：

```powershell
.\venv\Scripts\python.exe -X utf8 scripts\rebuild_milvus_index.py --check
.\venv\Scripts\python.exe -X utf8 scripts\rebuild_milvus_index.py --yes
```

### 4. 启动文本 API

```powershell
.\venv\Scripts\python.exe -m uvicorn app.service_api:app `
  --host 127.0.0.1 --port 8000 --workers 1
```

验证：

```powershell
curl.exe http://127.0.0.1:8000/health/live
curl.exe http://127.0.0.1:8000/health/ready
```

Swagger：<http://127.0.0.1:8000/docs>

### 5. 可选：启动多模态图片 API

视觉服务使用独立虚拟环境，避免 PaddleOCR 与主服务依赖冲突：

```powershell
py -3.10 -m venv services\vision\.venv
services\vision\.venv\Scripts\python.exe -m pip install `
  -r services\vision\requirements.txt
.\scripts\11_start_vision_api.ps1
```

Swagger：<http://127.0.0.1:8011/docs>。调用方法见
[services/vision/API.md](services/vision/API.md)。

## 测试

主项目：

```powershell
.\venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

视觉模块：

```powershell
.\services\vision\.venv\Scripts\python.exe -X utf8 `
  -m unittest discover -s tests -p "test_vision*.py" -v
```

## 目录说明

```text
app/                 FastAPI、对话服务和反馈存储
rag/                 Milvus、混合检索、重排与 RAG 评测
services/vision/     OCR、VLM、证据融合和图片 API
integrations/        飞书机器人接入
evaluation/          可复用评测代码；私有样本和逐条输出不公开
deployment/          Ollama 与服务部署模板
scripts/             启动、索引构建和环境检查脚本
tests/               单元与回归测试
docs/                架构、接口、评测和发布说明
```

## 安全与数据边界

- 复制 `.env.example` 生成 `.env`，不要在模板中填写真实密钥。
- API 默认只监听 `127.0.0.1`；对外提供服务时应使用 HTTPS、反向代理、API Key 和防火墙。
- 原始知识库、评测问题、逐条生成结果、模型文件和数据库均被 `.gitignore` 排除。
- 公开仓库不提供任何训练、验证、测试、holdout 或评测结果文件。
- 公开前请阅读 [docs/PUBLIC_RELEASE.md](docs/PUBLIC_RELEASE.md)。

## License

当前仓库暂未附带开源许可证。在代码归属和公开授权确认前，请勿假定可以复制、
再分发或用于商业用途。
