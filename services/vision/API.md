# EnBot 多模态诊断 API

这是独立于文本 RAG（8000 端口）的图片诊断服务。它在 8011 端口接收图片，依次执行图片校验、OCR、Qwen3-VL 结构化识别、证据融合，并仅在确认图片属于 EnValt/Hanchu 产品时调用现有 EnBot RAG。

## 启动前提

1. Ollama 已运行，且 `ollama list` 中存在 `qwen3-vl:2b-instruct`。
2. EnBot 文本 API 已运行在 `http://127.0.0.1:8000`。
3. 已安装视觉服务依赖：

```powershell
D:\telebot\services\vision\.venv\Scripts\python.exe -m pip install `
  -r D:\telebot\services\vision\requirements.txt `
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 启动服务

在 PowerShell 中执行：

```powershell
Set-Location D:\telebot
.\scripts\11_start_vision_api.ps1
```

默认只允许本机访问：

- Swagger：`http://127.0.0.1:8011/docs`
- 存活检查：`http://127.0.0.1:8011/health/live`
- 就绪检查：`http://127.0.0.1:8011/health/ready`

如需局域网演示，明确确认 Windows 防火墙规则后再使用：

```powershell
.\scripts\11_start_vision_api.ps1 -HostAddress 0.0.0.0 -Port 8011
```

## Postman 调用

新建 `POST http://127.0.0.1:8011/v1/diagnose`，Body 选择 `form-data`：

| Key | 类型 | 必填 | 示例 |
| --- | --- | --- | --- |
| `image` | File | 是 | 设备照片或接线图 |
| `question` | Text | 否 | `请分析图片中的设备和接线信息` |
| `session_id` | Text | 否 | `postman-demo` |
| `debug` | Text | 否 | `false` |

支持 PNG、JPEG、WEBP、BMP、TIFF；默认最大 10 MiB。不要手动设置 `Content-Type`，Postman 会自动生成带 boundary 的 multipart 请求头。

如果配置了环境变量 `VISION_API_KEY`，还需在 Headers 中加入：

```text
X-API-Key: 你的密钥
```

成功响应示例：

```json
{
  "status": "success",
  "request_id": "...",
  "data": {
    "status": "success",
    "answer": "...",
    "sources": [],
    "retrieval_backend": "milvus",
    "fusion": {},
    "total_seconds": 6.25,
    "stage_seconds": {
      "preprocessing": 0.2,
      "ocr": 2.3,
      "vision": 3.4,
      "fusion": 0.01,
      "rag": 1.1
    }
  }
}
```

若识别到外部品牌，响应状态为 `out_of_domain`，并且不会调用 EnBot 知识库。

## PowerShell 调用

```powershell
curl.exe -X POST "http://127.0.0.1:8011/v1/diagnose" `
  -F "image=@D:\path\device.png" `
  -F "question=请分析图片中的设备和接线信息" `
  -F "session_id=vision-demo" `
  -F "debug=false"
```

## 数据与安全边界

- 上传文件写入 `D:\telebot\tmp\vision-uploads`，请求结束后自动删除。
- 预处理缓存写入 `D:\telebot\tmp\vision-preprocessed`。
- Paddle 模型写入 `D:\AI-Cache\paddle`。
- `debug` 默认禁用；只有设置 `VISION_ENABLE_DEBUG=True` 后才允许返回 OCR/VLM 原始结构。
- `VISION_MAX_CONCURRENCY=1` 防止 16 GiB 显存被并发请求占满。
