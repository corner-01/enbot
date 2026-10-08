# EnBot 用户反馈机制

## 数据流

1. `/v1/chat` 返回回答和唯一的 `request_id`。
2. 问答内容自动写入 `data/feedback/enbot_feedback.db`。
3. 用户通过 `/v1/feedback` 对该 `request_id` 点赞或点踩。
4. 点踩记录进入 `pending` 人工复核；点赞记录标记为 `accepted`。
5. 反馈只用于复核和数据分析，不会自动进入训练集。

会话标识在数据库中只保存 SHA-256 哈希值，不保存原始飞书 `chat_id`。

## 提交反馈

```http
POST /v1/feedback
Content-Type: application/json
X-API-Key: <仅在服务配置了 ENBOT_API_KEY 时填写>
```

```json
{
  "request_id": "从问答响应中复制",
  "rating": "down",
  "categories": ["回答不完整"],
  "comment": "缺少设置完成后的确认步骤",
  "expected_answer": "可选的正确答案"
}
```

允许的错误类别：事实错误、回答不完整、回答不相关、型号混淆、数值或单位错误、
链接或页码错误、应拒答但没有拒答、错误拒答、其他。

## 管理接口

- `GET /v1/feedback/stats`：反馈数量、满意率、待复核数量和错误类别统计。
- `GET /v1/feedback/export?limit=1000`：导出反馈及对应问题和回答。

管理接口与问答接口使用相同的 `X-API-Key` 保护。

## 飞书使用

飞书启用 RAG 后，每条回答末尾会显示反馈提示。用户可以直接回复：

- `有帮助`
- `没帮助：缺少操作步骤`
- `👍` 或 `👎`

反馈会自动绑定到该飞书会话最近一次成功回答。可通过
`FEISHU_FEEDBACK_ENABLED=false` 关闭飞书反馈提示。
