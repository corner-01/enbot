# EnBot：从 Chroma 迁移到 Milvus

## 为什么升级

Chroma 适合单机原型和小规模调试，但 EnBot 已进入 M3：需要持续更新知识库、做混合检索实验，并准备 FastAPI/飞书多人演示。Milvus 以独立服务运行，集合、索引、元数据过滤、并发访问和后续扩容更适合这个阶段；同一客户端接口还可以从 Standalone 迁移到 Distributed。

需要说明：**更换向量数据库本身不会自动提高答案准确率**。准确率主要来自切分、Embedding、BM25、RRF、重排和证据约束。升级 Milvus 的价值是让这些检索实验具备更稳定、可扩展的工程底座。

## 本次代码改动

1. `VECTOR_STORE=milvus` 时使用 Milvus，连接地址和集合名全部来自环境变量。
2. 保留 `VECTOR_STORE_FALLBACK=chroma`，Milvus未启动时本地演示仍可回退。
3. BM25直接使用源文档切片，不再调用 Chroma 专属的 `get()`，因此两种后端共用同一套 BM25 + RRF + CPU重排逻辑。
4. 建库时只计算一次向量，移除了旧代码“先手工向量化、再由向量库重复向量化”的浪费。
5. 新增独立的检查/重建命令，普通API启动只加载已有集合，不会无提示删除索引。

## Windows 本机部署步骤

Milvus Standalone 运行在 Docker 中。默认把 compose 文件和持久化卷放到 `D:\enbot-milvus`，避免项目数据继续占用 C 盘。

### 1. 启动 Docker Desktop

确认 Docker Desktop 已启动并使用 Linux/WSL 2 引擎。若 Docker 的虚拟磁盘仍放在 C 盘，请先在 Docker Desktop 设置中把 Disk image location 调整到 D 盘。

### 2. 启动 Milvus

在项目 PowerShell 中执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\09_start_milvus.ps1
```

启动后：

- Milvus：`http://127.0.0.1:19530`
- WebUI：`http://127.0.0.1:9091/webui/`
- 数据：`D:\enbot-milvus\volumes`

### 3. 安装依赖

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. 配置 `.env`

把以下内容加入本地 `.env`，不要提交密钥或 `.env`：

```dotenv
VECTOR_STORE=milvus
VECTOR_STORE_FALLBACK=chroma
MILVUS_URI=http://127.0.0.1:19530
MILVUS_TOKEN=
MILVUS_DB_NAME=default
MILVUS_COLLECTION=enbot_knowledge_v1
MILVUS_CONSISTENCY_LEVEL=Session
MILVUS_INSERT_BATCH_SIZE=32
```

`KNOWLEDGE_FILE` 和 `EMBEDDING_MODEL_PATH` 继续使用现有配置。切换 Embedding 模型、切分规则或知识库内容后，应使用新集合名或重新建库。

### 5. 检查连接

```powershell
.\venv\Scripts\python.exe -X utf8 .\scripts\rebuild_milvus_index.py --check
```

首次运行显示 `Collection exists: False` 是正常的。

### 6. 首次建库或明确重建

下面的命令会删除同名集合后重新向量化，因此只有确认配置正确后才执行：

```powershell
.\venv\Scripts\python.exe -X utf8 .\scripts\rebuild_milvus_index.py --yes
```

建库完成后再次执行 `--check`，应看到集合存在以及行数统计。

### 7. 启动并验证 EnBot

```powershell
.\venv\Scripts\python.exe -X utf8 main.py
```

另开终端执行：

```powershell
.\venv\Scripts\python.exe -X utf8 .\evaluation\retrieval_benchmark.py
```

也可以用 Postman 请求 `POST http://127.0.0.1:8000/chat`，将 `debug` 设为 `true` 查看召回结果。

## 回滚

Milvus出现问题时只需修改：

```dotenv
VECTOR_STORE=chroma
```

原来的 `chroma_db` 不会在迁移过程中被删除；确认 Milvus评测稳定后再决定是否归档旧索引。
