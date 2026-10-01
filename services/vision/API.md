# Vision API

创建 `POST http://127.0.0.1:8011/v1/diagnose` 请求，Body 使用 `multipart/form-data`：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `image` | File | 是 | 待分析图片 |
| `question` | Text | 是 | 用户问题 |
| `session_id` | Text | 否 | 会话标识 |
| `debug` | Text | 否 | 默认 `false` |

若配置了 `VISION_API_KEY`，请求头还需提供 `X-API-Key`。支持的格式与大小限制以
`services/vision/app.py` 中的运行配置为准。公开仓库不附带真实图片和接口响应。
