# Vision service

该模块把图片校验、预处理、PaddleOCR、Ollama 视觉模型、证据融合和文本 RAG 串成一条
本地多模态诊断链路。

## 设计原则

- OCR 负责精确文字、型号、数值和单位；
- 视觉模型负责图片类型、状态、部件关系和结构化描述；
- 融合层不会用视觉模型内容覆盖 OCR 的精确事实；
- 无 OCR 支持的型号、厂商、故障码或故障描述会降级或移除；
- 域外图片不会调用知识库；
- 上传文件和预处理文件在请求结束后清理；
- OCR 使用独立 Python 环境，避免依赖冲突。

## 启动

```powershell
py -3.10 -m venv services\vision\.venv
services\vision\.venv\Scripts\python.exe -m pip install `
  -r services\vision\requirements.txt
.\scripts\11_start_vision_api.ps1
```

访问 <http://127.0.0.1:8011/docs> 查看 OpenAPI 文档。

## 主要环境变量

- `VISION_MODEL`：Ollama 中的视觉模型名称；
- `OLLAMA_BASE_URL`：Ollama 地址；
- `ENBOT_API_URL`：文本 RAG API 地址；
- `VISION_API_KEY`：可选接口密钥；
- `VISION_MAX_CONCURRENCY`：视觉模型并发数；
- `VISION_ENABLE_DEBUG`：是否允许返回调试证据。

本目录不包含任何图片样本、OCR 输出或知识库内容。
