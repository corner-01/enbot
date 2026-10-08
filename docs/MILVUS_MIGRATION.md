# EnBot：Milvus 迁移完成说明

EnBot 已完成从 Chroma 到 Milvus Standalone 的迁移。Milvus 现在是唯一支持的向量数据库，不再提供 Chroma 回退。

## 迁移原因

- 支持独立服务与持久化，便于多人访问和服务器部署；
- 支持 Collection、HNSW 索引、元数据过滤和后续扩容；
- 便于持续更新知识库和开展检索参数实验；
- 与 BM25、RRF 和 CPU 重排组成统一混合检索链路。

更换数据库本身不会自动提高准确率。当前效果主要来自数据切分、Embedding、BM25、RRF、重排、拒答阈值和独立评测集共同作用。

## 当前配置

```dotenv
VECTOR_STORE=milvus
MILVUS_URI=http://127.0.0.1:19530
MILVUS_DB_NAME=default
MILVUS_COLLECTION=enbot_knowledge_v1
MILVUS_CONSISTENCY_LEVEL=Session
MILVUS_INSERT_BATCH_SIZE=32
```

## Windows 本地启动

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\09_start_milvus.ps1
.\venv\Scripts\python.exe -X utf8 .\scripts\rebuild_milvus_index.py --check
```

需要重建 Collection 时：

```powershell
.\venv\Scripts\python.exe -X utf8 .\scripts\rebuild_milvus_index.py --yes
```

`--yes` 会删除并重建当前 Collection，执行前必须确认知识库、Embedding 模型和集合名称。

## 验证结论

- Holdout V2：56/60，通过率 93.33%，无答案识别率 100%；
- 最新检索回归：80/80；
- Milvus 已通过 API、飞书和回答质量 A/B 链路验证。

旧的本地 Chroma 索引和运行时依赖已经删除。
