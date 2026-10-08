# EnBot API 调用文档

## 1. 服务信息

- 服务名称：EnBot 家庭能源智能客服 API
- 当前版本：3.1.0
- 本地基础地址：`http://127.0.0.1:8000`
- 在线接口文档：`http://127.0.0.1:8000/docs`
- 返回格式：JSON
- 问答方式：非流式，一次请求返回完整答案
- 正式问答模型：`enbot-optuna:7b`
- 检索后端：Milvus 混合检索与本地重排

调用前应确保 Ollama、Milvus 和 EnBot API 已启动。Redis 用于持久保存会话历史，
不可用时服务会自动退回进程内存。

## 2. 鉴权

如果服务端 `.env` 中配置了 `ENBOT_API_KEY`，所有 `/v1/*` 业务接口都必须携带：

```http
X-API-Key: <服务端分配的密钥>
```

如果 `ENBOT_API_KEY` 为空，本地调用可以不传该请求头。正式服务器必须配置密钥。
不要把 API Key 写进公开网页、公开仓库或浏览器前端源码，应由后端代理或安全配置注入。

通用请求头：

```http
Content-Type: application/json
X-API-Key: <可选>
X-Request-ID: <可选的调用方追踪编号>
```

未提供 `X-Request-ID` 时，EnBot 会自动生成唯一编号，并同时放在响应头和响应正文中。

## 3. 问答接口

### `POST /v1/chat`

使用微调模型和 Milvus RAG 回答问题。

请求示例：

```json
{
  "query": "怎么设置负电价限制馈网？",
  "session_id": "frontend-user-001",
  "evaluate": false,
  "debug": false
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `query` | string | 是 | 用户问题，去除空格后长度为 1～8000 |
| `session_id` | string | 否 | 会话标识，默认 `default`，用于隔离不同用户的上下文 |
| `evaluate` | boolean | 否 | 是否运行在线 RAGAS，默认 `false`，生产环境应保持关闭 |
| `ground_truth` | string/null | 否 | 仅供评测时使用的参考答案 |
| `debug` | boolean | 否 | 是否返回检索调试信息，生产环境应保持关闭 |

`session_id` 只能包含英文字母、数字及 `_.:@-`，最长 128 个字符。前端可使用自身的
会话 ID，但不要直接传递手机号、邮箱等敏感信息。

成功响应：

```json
{
  "status": "success",
  "request_id": "fa9d3802ad5a40a4aa2c8f14df1034fb",
  "data": {
    "response": "打开 App，进入 MENU 页面……",
    "sources": [],
    "retrieval_backend": "milvus"
  }
}
```

字段说明：

| 字段 | 说明 |
|---|---|
| `request_id` | 本次回答的唯一标识，提交用户反馈时必须使用 |
| `data.response` | 最终回答文本 |
| `data.sources` | 知识条目包含的可展示来源链接；为空不代表没有使用知识库 |
| `data.retrieval_backend` | 当前检索后端，正式环境应为 `milvus` |
| `data.ragas` | 仅在开启在线评测时返回 |
| `data.retrieval_debug` | 仅在允许调试并设置 `debug=true` 时返回 |

前端应至少保存最近一次回答的 `request_id`，用于后续点赞或点踩。

## 4. 独立检索接口

### `POST /v1/retrieve`

只运行知识库检索，不调用大模型生成答案，适合调试检索结果或制作内部工具。

请求示例：

```json
{
  "query": "HESS-HY-T-4~20K-M 电网数据显示异常",
  "include_content": false
}
```

成功响应：

```json
{
  "status": "success",
  "request_id": "6f61cc6e0fe44cf49d86808e90e6f639",
  "data": {
    "retrieval_backend": "milvus",
    "retrieval_debug": [
      {
        "rank": 1,
        "source": "售后.json",
        "row": 10,
        "reranker_score": 0.99
      }
    ]
  }
}
```

`include_content=true` 会暴露知识库正文，因此只有服务端配置
`ENBOT_ENABLE_DEBUG=true` 时才允许使用。

## 5. 用户反馈接口

### `POST /v1/feedback`

反馈必须绑定 `/v1/chat` 返回的 `request_id`。

点赞示例：

```json
{
  "request_id": "fa9d3802ad5a40a4aa2c8f14df1034fb",
  "rating": "up",
  "categories": [],
  "comment": "",
  "expected_answer": ""
}
```

点踩示例：

```json
{
  "request_id": "fa9d3802ad5a40a4aa2c8f14df1034fb",
  "rating": "down",
  "categories": ["回答不完整"],
  "comment": "缺少设置完成后的确认步骤",
  "expected_answer": "可选：人工提供的正确答案"
}
```

允许的 `categories`：

- `事实错误`
- `回答不完整`
- `回答不相关`
- `型号混淆`
- `数值或单位错误`
- `链接或页码错误`
- `应拒答但没有拒答`
- `错误拒答`
- `其他`

成功响应：

```json
{
  "status": "success",
  "data": {
    "id": 1,
    "request_id": "fa9d3802ad5a40a4aa2c8f14df1034fb",
    "rating": "down",
    "review_status": "pending",
    "created_at": "2026-09-30T08:00:00+00:00",
    "updated_at": "2026-09-30T08:00:00+00:00"
  }
}
```

同一个 `request_id` 再次提交反馈时会更新原记录，不会生成重复反馈。点赞记录状态为
`accepted`，点踩记录状态为 `pending`。反馈不会自动进入训练集。

### `GET /v1/feedback/stats`

返回反馈总数、点赞数、点踩数、满意率、待复核数量和错误类别统计。

### `GET /v1/feedback/export?limit=1000`

导出反馈及其对应问题、回答、来源和耗时。该接口包含业务数据，只能提供给内部管理端。

## 6. 健康检查与指标

### `GET /health/live`

只检查 API 进程是否存活，不加载或验证模型。

### `GET /health/ready`

检查 API、Milvus 检索器和 Ollama 模型是否全部可用。成功时返回 HTTP 200 和：

```json
{
  "status": "ready",
  "checks": {
    "runtime_initialized": true,
    "retrieval_backend": "milvus",
    "milvus_active": true,
    "ollama": {
      "available": true,
      "expected_model": "enbot-optuna:7b"
    }
  }
}
```

服务未准备完成时返回 HTTP 503，`detail.checks` 会指出缺少的组件。

### `GET /v1/metrics`

返回运行时间、会话数、请求总数、失败数和平均请求耗时。该接口需要 API Key。

## 7. Postman 调用

1. 新建 `POST` 请求。
2. 地址填写 `http://127.0.0.1:8000/v1/chat`。
3. 在 `Headers` 添加 `Content-Type: application/json`。
4. 如果配置了密钥，再添加 `X-API-Key`。
5. 选择 `Body` → `raw` → `JSON`。
6. 粘贴问答请求 JSON 后点击 `Send`。
7. 保存响应里的 `request_id`，再调用 `/v1/feedback` 提交反馈。

## 8. JavaScript 调用示例

```javascript
async function askEnBot(query, sessionId) {
  const response = await fetch("http://127.0.0.1:8000/v1/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
      // "X-API-Key": "由安全后端代理添加"
    },
    body: JSON.stringify({
      query,
      session_id: sessionId,
      evaluate: false,
      debug: false
    })
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || `HTTP ${response.status}`);
  }
  return payload;
}
```

提交反馈：

```javascript
async function submitFeedback(requestId, rating, comment = "") {
  const response = await fetch("http://127.0.0.1:8000/v1/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      request_id: requestId,
      rating,
      categories: [],
      comment,
      expected_answer: ""
    })
  });
  return await response.json();
}
```

前端和 API 不同域名时，需要服务端通过 `ENBOT_CORS_ORIGINS` 明确配置允许的前端来源。

## 9. Python 调用示例

```python
import requests

base_url = "http://127.0.0.1:8000"
headers = {
    "Content-Type": "application/json",
    # "X-API-Key": "服务端分配的密钥",
}

response = requests.post(
    f"{base_url}/v1/chat",
    headers=headers,
    json={
        "query": "怎么设置负电价限制馈网？",
        "session_id": "python-demo",
        "evaluate": False,
        "debug": False,
    },
    timeout=300,
)
response.raise_for_status()
payload = response.json()
print(payload["data"]["response"])
print("request_id:", payload["request_id"])
```

## 10. 常见状态码

| HTTP 状态码 | 含义 | 处理方式 |
|---:|---|---|
| 200 | 调用成功 | 正常读取 JSON |
| 401 | API Key 缺失或错误 | 检查 `X-API-Key` |
| 403 | 在线评测或调试功能未开放 | 将 `evaluate/debug` 设为 `false` |
| 404 | 反馈使用了不存在的 `request_id` | 先调用新版问答接口取得编号 |
| 422 | JSON 字段或格式不合法 | 检查必填字段、类型和 `session_id` |
| 503 | 模型、Milvus 或服务尚未准备好 | 调用 `/health/ready` 查看具体原因 |

## 11. 调用注意事项

- 前端超时时间建议设置为 300 秒，本地首次加载模型可能较慢。
- 当前单机 16 GB GPU 配置的默认并发数为 1，前端应显示“处理中”状态并避免重复提交。
- 同一用户连续对话应持续使用同一个 `session_id`；不同用户不能共用。
- `sources=[]` 不代表未检索知识库，只表示当前知识条目没有可返回的 URL。
- 正式调用优先使用 `/v1/chat` 和 `/v1/retrieve`。
- `/chat` 和 `/retrieve` 仅用于兼容旧版 Postman、飞书或客户端。
- 生产环境不要开启 `debug` 和在线 RAGAS，以免暴露知识正文或增加延迟。
