# EnBot API

启动命令：

```powershell
.\venv\Scripts\python.exe -m uvicorn app.service_api:app `
  --host 127.0.0.1 --port 8000 --workers 1
```

启动后访问 <http://127.0.0.1:8000/docs> 查看以代码为准的 OpenAPI 文档。

主要接口：

- `GET /health/live`：进程存活检查；
- `GET /health/ready`：模型、检索后端等依赖的就绪检查；
- `POST /v1/chat`：文本问答；
- `POST /v1/feedback`：记录用户对回答是否有帮助的反馈。

生产环境请配置 API Key、请求大小限制、日志脱敏、HTTPS 和反向代理。公开仓库不提供
任何真实请求、响应、知识库片段或用户反馈记录。
