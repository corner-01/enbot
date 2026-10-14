# EnBot SSE与Feedback接口对接说明

## 1. SSE聊天接口

### 1.1 接口地址

```http
POST http://192.168.3.165:8000/v1/chat/stream
Content-Type: application/json
Accept: text/event-stream
X-API-Key: <后端启用鉴权时必填>
```

这是`POST`请求，请使用`fetch`读取响应流。

### 1.2 请求格式

```json
{
  "query": "用户问题",
  "session_id": "user-001"
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `query` | string | 用户问题，不能为空 |
| `session_id` | string | 会话编号；同一段对话保持一致，不同用户不要共用 |

### 1.3 事件顺序

正常情况：

```text
meta → heartbeat（可能没有）→ delta（多次）→ sources → done
```

异常情况：

```text
meta → error
```

| 事件 | 处理方式 |
|---|---|
| `meta` | 保存`request_id`，确认连接已建立 |
| `heartbeat` | 后端仍在处理，继续等待 |
| `delta` | 按`index`顺序将`delta`追加到当前回答 |
| `sources` | 读取知识来源和`retrieval_backend` |
| `done` | 请求成功结束 |
| `error` | 请求失败，结束本次请求 |

请保存`meta`事件中的`request_id`，后续提交Feedback时使用。

### 1.4 联调验收

- 能够建立`text/event-stream`连接。
- 能够收到`meta`事件和`request_id`。
- 多个`delta`能够按顺序拼接。
- 能够收到`sources`和`done`。
- 发生异常时能够正确处理`error`。
- 长时间处理时不会误判连接失败。

当前接口会先完成RAG检索和回答生成，再通过多个`delta`事件分片发送；不是模型逐Token实时输出。

## 2. Feedback接口

### 2.1 接口地址

```http
POST http://192.168.3.165:8000/v1/feedback
Content-Type: application/json
X-API-Key: <后端启用鉴权时必填>
```

`request_id`取自SSE聊天接口的`meta`事件。

### 2.2 请求参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `request_id` | string | 是 | 对应聊天回答的请求编号 |
| `rating` | string | 是 | `up`表示有帮助，`down`表示没帮助 |
| `categories` | string[] | 否 | 没帮助的原因 |
| `comment` | string | 否 | 用户补充说明 |
| `expected_answer` | string | 否 | 期望的正确答案 |

### 2.3 有帮助请求示例

```json
{
  "request_id": "fa9d3802ad5a40a4aa2c8f14df1034fb",
  "rating": "up",
  "categories": [],
  "comment": "",
  "expected_answer": ""
}
```

### 2.4 没帮助请求示例

```json
{
  "request_id": "fa9d3802ad5a40a4aa2c8f14df1034fb",
  "rating": "down",
  "categories": ["回答不完整"],
  "comment": "缺少具体操作步骤",
  "expected_answer": ""
}
```

`categories`可选值：

```text
事实错误
回答不完整
回答不相关
型号混淆
数值或单位错误
链接或页码错误
应拒答但没有拒答
错误拒答
其他
```

### 2.5 成功响应

```json
{
  "status": "success",
  "data": {
    "id": 1,
    "request_id": "fa9d3802ad5a40a4aa2c8f14df1034fb",
    "rating": "down",
    "review_status": "pending",
    "created_at": "2026-10-10T08:00:00+00:00",
    "updated_at": "2026-10-10T08:00:00+00:00"
  }
}
```

### 2.6 错误响应

| HTTP状态码 | 说明 |
|---|---|
| `401` | API Key缺失或错误 |
| `404` | `request_id`不存在 |
| `422` | 请求参数不正确 |
| `503` | Feedback功能未启用 |
