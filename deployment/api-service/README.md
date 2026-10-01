# EnBot API 服务部署

生产候选固定为 **Optuna QLoRA 模型 `enbot-optuna:7b` + Milvus 混合检索/RAG**。
DPO 试验没有通过开发集门槛，因此不进入本服务。

## 1. 服务组成

- EnBot FastAPI：默认 `127.0.0.1:8000`
- Ollama：`127.0.0.1:11434`，模型 `enbot-optuna:7b`
- Milvus：`127.0.0.1:19530`，集合 `enbot_knowledge_v1`
- Redis：`127.0.0.1:6379`，不可用时自动退化为进程内存

API默认只启动一个进程和一个并发推理槽。不要增加Uvicorn worker数量，否则每个worker都会加载一份Embedding模型并重复占用GPU显存。

## 2. 配置

在项目根目录执行：

```bash
cp deployment/api-service/.env.production.example .env
nano .env
```

至少修改：

```dotenv
ENBOT_API_KEY=替换为足够长的随机密钥
EMBEDDING_MODEL_PATH=./models/Qwen3-8B-native
RERANKER_MODEL_PATH=./models/bge-reranker-v2-m3
KNOWLEDGE_FILE=./knowledge_data
```

如果只通过SSH隧道或Nginx访问，保留 `ENBOT_API_HOST=127.0.0.1`。只有在受保护的内网中才改成 `0.0.0.0`，并且必须设置API密钥和防火墙。

## 3. 启动依赖

```bash
sudo systemctl start ollama redis-server docker
ollama list
redis-cli ping
docker ps --filter name=milvus-standalone
```

必须看到Ollama中存在 `enbot-optuna:7b`，Milvus容器状态为healthy。

## 4. 手动启动与验收

```bash
cd /opt/enbot
bash deployment/api-service/start_api.sh
```

第一次启动会加载本地Embedding模型、重排模型和Milvus集合，需要等待 `/health/ready` 通过：

```bash
curl -fsS http://127.0.0.1:8000/health/live
curl -fsS http://127.0.0.1:8000/health/ready
```

调用聊天接口：

```bash
curl -fsS -X POST http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: 替换为.env中的ENBOT_API_KEY' \
  -d '{"query":"怎么设置负电价限制馈网？","session_id":"demo-user"}'
```

检索接口：

```bash
curl -fsS -X POST http://127.0.0.1:8000/v1/retrieve \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: 替换为.env中的ENBOT_API_KEY' \
  -d '{"query":"示例设备的状态数据异常"}'
```

Swagger接口文档位于 `http://127.0.0.1:8000/docs`。旧的 `/chat` 和 `/retrieve` 路径继续保留，飞书程序无需立即修改。

## 5. Postman

1. Method选择 `POST`，URL填写 `http://服务器地址:8000/v1/chat`。
2. Headers添加 `Content-Type: application/json`。
3. 如果设置了API密钥，再添加 `X-API-Key: 密钥`。
4. Body选择 `raw` 和 `JSON`：

```json
{
  "query": "怎么设置负电价限制馈网？",
  "session_id": "postman-demo",
  "debug": false,
  "evaluate": false
}
```

## 6. 注册systemd服务

复制模板并替换用户与项目路径：

```bash
sudo cp deployment/api-service/enbot-api.service.example /etc/systemd/system/enbot-api.service
sudo nano /etc/systemd/system/enbot-api.service
sudo systemctl daemon-reload
sudo systemctl enable --now enbot-api
sudo systemctl status enbot-api --no-pager
sudo journalctl -u enbot-api -n 100 --no-pager
```

## 7. 接口说明

- `GET /health/live`：进程存活检查，不代表模型可用。
- `GET /health/ready`：检查运行时、Milvus后端和Ollama模型。
- `POST /v1/chat`：生产聊天接口，支持 `session_id` 会话隔离。
- `POST /v1/retrieve`：仅检索，不调用回答模型。
- `GET /v1/metrics`：服务请求数、失败数、平均耗时和会话数。
- `POST /chat`、`POST /retrieve`：兼容原Postman和飞书调用。

生产环境默认关闭 `debug` 和在线RAGAS评测，避免泄露知识库正文或让单次请求占用过多时间。需要排障时可临时设置 `ENBOT_ENABLE_DEBUG=true`，完成后立即关闭。
