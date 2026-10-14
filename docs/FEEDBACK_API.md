# Feedback 接口文档

## 接口信息

```http
POST http://192.168.3.165:8000/v1/feedback
Content-Type: application/json
```

`request_id`取自聊天接口返回结果。

## 请求参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `request_id` | string | 是 | 对应聊天回答的请求编号 |
| `rating` | string | 是 | `up`表示有帮助，`down`表示没帮助 |
| `categories` | string[] | 否 | 没帮助的原因 |
| `comment` | string | 否 | 用户补充说明 |
| `expected_answer` | string | 否 | 期望的正确答案 |

## 有帮助请求示例

```json
{
  "request_id": "fa9d3802ad5a40a4aa2c8f14df1034fb",
  "rating": "up",
  "categories": [],
  "comment": "",
  "expected_answer": ""
}
```

## 没帮助请求示例

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

## 成功响应

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

## 错误响应

| HTTP状态码 | 说明 |
|---|---|
| `401` | API Key缺失或错误 |
| `404` | `request_id`不存在 |
| `422` | 请求参数不正确 |
| `503` | Feedback功能未启用 |
