# EnBot Gradio 演示页

该页面只调用已经运行的 EnBot 文本 API 和视觉 API，不会重复加载模型。

安装可选依赖：

```powershell
cd D:\telebot
.\venv\Scripts\python.exe -m pip install -r .\demo\requirements.txt
```

依次启动 Milvus、Ollama、文本 API、视觉 API后，运行：

```powershell
.\scripts\12_start_gradio_demo.ps1
```

浏览器打开 `http://127.0.0.1:7860`。页面默认不创建公网分享链接。

如需在同一局域网内展示：

```powershell
.\scripts\12_start_gradio_demo.ps1 -HostAddress 0.0.0.0
```

其他电脑使用 `http://本机局域网IP:7860` 访问。仅应在可信局域网中使用，
并应配置 Windows 防火墙限制访问范围。
