# EnBot 飞书长连接

这个程序使用飞书官方 Python SDK 的长连接（WebSocket）接收消息，然后调用本机已经启动的 EnBot `/chat` 接口，并把回答回复到原飞书消息。它不会再次加载大模型、向量模型或 RAG，因此不会额外占用一套模型显存。

## 1. 配置凭证

在项目根目录 `.env` 中填写：

```env
FEISHU_APP_ID=飞书应用的App ID
FEISHU_APP_SECRET=飞书应用的App Secret
```

不要把真实凭证提交到 Git 仓库，也不要通过聊天发送 App Secret。

## 2. 飞书后台配置

1. 为应用启用“机器人”能力。
2. 在“权限管理”中申请机器人发送消息权限（`im:message:send_as_bot`）及接收消息所需权限。
3. 在“事件与回调”中选择“使用长连接接收事件”。
4. 添加事件“接收消息 `im.message.receive_v1`”。
5. 先启动下面的长连接程序；终端显示连接成功后，再回到飞书后台点击“验证”并保存。
6. 创建并发布应用版本，把机器人加入测试群或设置测试用户。

## 3. 启动顺序

先启动 Ollama 和 EnBot API，确认 `http://127.0.0.1:8000/` 能访问。然后在 PowerShell 中运行：

```powershell
cd D:\telebot
.\scripts\07_start_feishu_bot.ps1
```

保持该终端窗口运行。按 `Ctrl+C` 停止长连接程序。

## 4. 自检命令

只检查程序逻辑，不连接飞书：

```powershell
D:\telebot\venv\Scripts\python.exe D:\telebot\integrations\feishu_bot.py --self-test
```

检查飞书配置是否已填写以及 EnBot API 是否可访问：

```powershell
D:\telebot\venv\Scripts\python.exe D:\telebot\integrations\feishu_bot.py --check
```

## 5. 可选配置

这些配置已在 `.env.example` 中给出默认值：

- `ENBOT_API_URL`：EnBot 聊天接口地址。
- `FEISHU_ENBOT_TIMEOUT`：等待本地模型回答的最长秒数。
- `FEISHU_QUEUE_SIZE`：同时等待处理的消息数上限。
- `FEISHU_MAX_REPLY_CHARS`：单条飞书回复最大字符数，超出后自动分段。
- `FEISHU_REQUIRE_GROUP_MENTION=true`：群聊中只有 @小英 才响应；私聊直接响应。
- `FEISHU_ALLOWED_CHAT_IDS`：可选的会话 ID 白名单，多个 ID 用英文逗号分隔；留空代表不限制。

当前演示版只处理文本消息。图片、文件等消息会收到“目前只支持文本消息”的提示。

## 6. 飞书直连微调模型（不使用 RAG）

先停止正在运行的飞书长连接（按 `Ctrl+C`），确认 Ollama 中能看到
`enbot-optuna:7b`，然后运行：

```powershell
cd D:\telebot
.\scripts\08_start_feishu_direct_model.ps1
```

该脚本会设置 `FEISHU_BACKEND=ollama`，飞书消息将直接发送到 Ollama
`/api/chat`。此模式不需要启动 `main.py`、Milvus、Embedding、重排器或
Redis。要恢复 RAG 模式，停止它并重新运行：

```powershell
.\scripts\07_start_feishu_bot.ps1
```
