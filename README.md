# enbot

enbot 是一个完全本地运行的储能设备智能客服系统，助手名称为“小英”。系统使用 Chroma 保存知识向量，结合 Qwen3 Embedding、BM25、RRF 和 CPU Reranker 完成检索，再由本地 Ollama 模型生成跨语言答案，并可使用 RAGAS 进行评测。

## 目录结构

```text
enbot/
├─ main.py                         # 唯一服务启动入口
├─ requirements.txt                # Python 依赖
├─ .env                            # 本地环境配置，不提交版本库
├─ app/
│  ├─ api.py                       # FastAPI 接口、助手“小英”和回答流程
│  └─ general_tools.py             # 搜索、天气、时间、计算器工具
├─ rag/
│  ├─ knowledge_retriever.py       # Embedding、Chroma 和检索器初始化
│  ├─ hybrid_search.py             # BM25、RRF、CPU 精排及型号保护
│  └─ evaluator.py                 # 本地 RAGAS 六项指标评测
├─ memory/
│  └─ redis_history.py             # Redis 对话历史与内存降级
├─ evaluation/
│  ├─ retrieval_benchmark.py       # 通过 API 执行检索回归测试
│  └─ retrieval_cases.json         # 检索测试问题和预期结果
├─ scripts/
│  ├─ check_gpu.py                 # 检查 PyTorch、CUDA 和显卡状态
│  └─ test_reranker.py             # 验证本地 CPU Reranker
├─ chroma_db/                      # 已构建的本地 Chroma 数据库
├─ models/                         # 本地 Embedding 和 Reranker 模型
└─ venv/                           # Python 虚拟环境
```

各包中的 `__init__.py` 只负责将目录声明为 Python 包，不包含业务逻辑。

## 文件职责

### `main.py`

项目唯一启动入口，加载 `app.api` 中的 FastAPI 应用并监听 `127.0.0.1:8000`。

### `app/api.py`

系统主流程，包括：

- `/chat`、根路由和 WebSocket 接口；
- 本地 Ollama `qwen2.5:7b` 初始化；
- 产品问题强制进入本地 RAG；
- 标准 Q&A 答案提取和跨语言整理；
- 来源链接分离；
- Redis 对话历史；
- RAGAS 评测和检索调试信息。

### `app/general_tools.py`

只保留 Agent 实际使用的四个通用工具：网络搜索、天气、当前时间和数学计算。产品技术问题不会优先调用网络搜索。

### `rag/knowledge_retriever.py`

负责加载本地 Qwen3 Embedding 和 Chroma，创建语义召回器，并将语义召回、BM25、RRF 和 Reranker 组合成最终知识检索器。

### `rag/hybrid_search.py`

负责检索精度优化：

- 中英文技术词分词和 BM25；
- 查询扩展和 RRF 融合；
- 本地 `bge-reranker-v2-m3` CPU 精排；
- Q&A 问题对问题匹配；
- 产品型号精确过滤和跨型号保护；
- 编号类问题结构校验；
- 纯图片链接过滤、动态阈值和结果缓存。

### `rag/evaluator.py`

使用本地 LLM 和本地 Embedding 执行 RAGAS 评测，输出 Faithfulness、Answer Relevancy、Context Precision、Context Recall、Answer Correctness 和 Answer Similarity。代码会阻止评测回退到付费 API。

### `memory/redis_history.py`

保存用户问题和系统回答，支持短期历史、长期历史和 Redis 不可用时的内存降级。短期数据默认保存 24 小时，长期数据默认保存 7 天。

### `evaluation/retrieval_benchmark.py`

调用正在运行的 `/chat` 接口，检查目标文档排名，计算通过率和 MRR。适合在新增知识、修改检索参数或微调 Reranker 后执行回归。

### `evaluation/retrieval_cases.json`

保存可维护的检索测试集，包括正常命中、图片资料和跨型号保护等用例。

### `scripts/check_gpu.py`

输出 PyTorch、CUDA、cuDNN 和 GPU 信息，用于环境诊断。

### `scripts/test_reranker.py`

在 CPU 上加载本地 Reranker，对相关与无关文本打分，确认模型和运行设备正确。

## 启动服务

```powershell
cd enbot
.\venv\Scripts\python.exe -X utf8 main.py
```

接口文档：`http://127.0.0.1:8000/docs`

Postman 示例：

```json
{
  "query": "HESS-HY-T-4~20K-M逆变器显示的电网侧数据不对",
  "evaluate": true,
  "ground_truth": "知识库标准答案",
  "debug": true
}
```

## 验证命令

```powershell
# 检索回归
.\venv\Scripts\python.exe -X utf8 evaluation\retrieval_benchmark.py

# CPU Reranker
.\venv\Scripts\python.exe -X utf8 scripts\test_reranker.py

# GPU 环境
.\venv\Scripts\python.exe -X utf8 scripts\check_gpu.py
```

## 可选环境变量

- `OLLAMA_MODEL`：本地 Ollama 模型名，默认是 `qwen2.5:7b`。
- `KNOWLEDGE_FILE`：首次构建知识库时的 Excel/TXT 目录。
- `CHROMA_PATH`：Chroma 持久化目录，默认是项目内的 `chroma_db`。
- `EMBEDDING_MODEL_PATH`：Embedding 模型目录，默认是项目内的 Qwen3 模型。
- `EMBEDDING_DEVICE`：Embedding 使用 `cuda` 或 `cpu`。
- `RERANKER_MODEL_PATH`：本地 Reranker 模型目录。
- `RERANKER_ENABLED`：是否启用 CPU Reranker。
- `RAGAS_DO_NOT_TRACK`：关闭 RAGAS 匿名遥测。
- `ENBOT_FINETUNE_ROOT`：可选的 WSL 微调工作区目录。

项目运行不需要 OpenAI、DashScope、SerpAPI 等付费 API 密钥；`.env` 中遗留的相关配置不会被当前代码读取。

## 上传仓库前检查

- 不要提交 `.env`、本地模型、Chroma 数据库、知识库原始文件、虚拟环境或客户资料。
- 仓库中只保留 `.env.example`，供其他开发者复制并填写自己的本地路径。
- 根据公司要求决定仓库是否必须设为私有；公开仓库前应完成数据和品牌信息审查。
- 当前项目尚未指定许可证。没有公司授权时，不要自行添加开源许可证。
- 仓库创建完成后，先检查待上传文件清单，再进行首次提交。
