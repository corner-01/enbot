# EnBot MCP 接入说明

EnBot MCP 是现有 FastAPI 服务的轻量适配层。它不会再次加载 Ollama、
Embedding、Reranker 或 Milvus，只把现有能力转换为 MCP 工具。

## 工具

| 工具 | 用途 |
| --- | --- |
| `enbot_health` | 检查 EnBot、Milvus 和 Ollama 是否就绪 |
| `enbot_ask` | 家庭储能问答，返回回答、来源和 `request_id` |
| `enbot_retrieve` | 只检索知识库，返回来源和相关度信息 |
| `enbot_feedback` | 使用问答的 `request_id` 提交有帮助/没帮助反馈 |

第一版没有设备控制工具，避免智能体未经人工确认执行充放电等高风险操作。

## 启动前提

先启动现有 EnBot API，并确认：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

首次使用需要安装新增依赖：

```powershell
py -3.10 -m venv venv-mcp
.\venv-mcp\Scripts\python.exe -m pip install -r integrations\requirements-mcp.txt
```

## 本机 stdio 接入

让支持 MCP 的客户端启动下面的命令：

```text
D:\telebot\venv-mcp\Scripts\python.exe -m integrations.mcp_server --transport stdio
```

客户端配置示例：

```json
{
  "mcpServers": {
    "enbot": {
      "command": "D:\\telebot\\venv-mcp\\Scripts\\python.exe",
      "args": ["-m", "integrations.mcp_server", "--transport", "stdio"],
      "cwd": "D:\\telebot",
      "env": {
        "ENBOT_MCP_API_BASE_URL": "http://127.0.0.1:8000"
      }
    }
  }
}
```

如果文本 API 设置了 `ENBOT_API_KEY`，请在客户端环境中同时设置
`ENBOT_MCP_API_KEY`，两者的值保持一致。

## 本机 Streamable HTTP 接入

```powershell
.\scripts\14_start_mcp_http.ps1
```

MCP 地址：`http://127.0.0.1:8020/mcp`

`GET /mcp` 不是普通网页，浏览器直接打开不代表工具是否正常；应使用 MCP
Inspector 或支持 MCP 的客户端连接并调用 `enbot_health`。

## 网络安全

默认只监听 `127.0.0.1`。不要直接把 8020 端口映射到公网。需要让局域网或
远程机器访问时，应先增加身份认证、HTTPS、访问白名单和反向代理，再把监听
地址改为 `0.0.0.0`。
