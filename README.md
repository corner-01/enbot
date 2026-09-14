# EnBot 服务器部署

本文档只说明现有版本的部署和资源预算。目标环境为 **Ubuntu 22.04 x86_64、NVIDIA GPU、Python 3.10**；下列命令在服务器终端执行，除特别注明外使用同一个普通登录账号。服务仅监听本机，不直接对公网开放。

## 一、服务器配置估算

**最低 GPU 显存：单卡 32 GB。** 按当前代码让回答模型和 Embedding 同时使用 GPU（`EMBEDDING_DEVICE=cuda`）、重排模型使用 CPU，按低并发部署计算。

**计算方法：**本地 Embedding 的四个 BF16 权重文件合计约 15.13 GB，Ollama `qwen2.5:7b` 的 Q4_K_M 模型文件约 4.7 GB，合计约 **19.83 GB**；再为 CUDA/PyTorch、Ollama 的 KV 缓存和临时计算预留约 **8–10 GB**，得到约 **28–30 GB**，取 **32 GB**。CPU 重排模型不占 GPU 显存。这是根据代码和模型文件大小算出的部署下限。

## 二、部署步骤

### 1. 取得代码和必须的私有文件

仓库**不包含** `.env`、模型权重、原始知识库和 `chroma_db`。部署人员需要有仓库读取权限，且要从项目负责人处取得**完整的** `models/Qwen3-8B-native/`、`models/bge-reranker-v2-m3/`，以及原始 TXT/XLSX 知识库（或已验证可用、用同一 Embedding 构建的完整 Chroma 目录）。不要把令牌写进 Git URL、README 或 `.env`。

```bash
sudo apt update
sudo apt install -y git
cd ~
git clone https://cnb.cool/envalt-tech/enbot
cd enbot
mkdir -p models knowledge_data
```

通过安全传输方式，把完整模型目录放到 `models/` 下，把 TXT/XLSX 放到 `knowledge_data/` 下；若知识库实际位于其他目录，稍后在 `.env` 中填写绝对路径。可选择复制现成 `chroma_db/` 以省去重建，但必须与当前 Embedding 模型匹配，并在 Linux 上通过实际问答验证。**要从原始文档重建时，不要预先创建空的 `chroma_db/` 目录**：当前代码只要发现该目录存在就会尝试直接加载，不会自动重建。

传输后先检查：

```bash
test -f models/Qwen3-8B-native/config.json
test -f models/Qwen3-8B-native/model.safetensors.index.json
test -f models/bge-reranker-v2-m3/model.safetensors
find knowledge_data -type f \( -name '*.txt' -o -name '*.xlsx' \) | head
```

若使用现成 Chroma，检查 `test -s chroma_db/chroma.sqlite3`；若从原始文档建库，确认上面 `find` 至少列出一个有效文件。模型目录必须包含 tokenizer、配置和全部权重分片，不是只拷贝一个 `.safetensors` 文件。

### 2. 安装 Python 与 GPU 依赖

服务器应先由供应商或管理员安装兼容的 NVIDIA 驱动，`nvidia-smi` 必须能显示 GPU。这里不提供驱动的通用安装命令，因为驱动版本取决于租用的镜像与显卡。

```bash
sudo apt update
sudo apt install -y curl python3.10 python3.10-venv python3-pip redis-server
nvidia-smi
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

在 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)选择 Linux / Pip / Python / 与服务器驱动兼容的 CUDA，**先执行页面给出的 PyTorch 安装命令**，再执行：

```bash
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

推荐的 GPU 档位必须看到 `CUDA: True`；否则先排查驱动和 PyTorch 安装，不能继续把 Embedding 设为 `cuda`。依赖文件使用版本范围，第一次验收通过后应保存实际安装版本以便复现。

### 3. 安装并启动 Ollama、Redis

按 [Ollama 官方 Linux 文档](https://docs.ollama.com/linux)安装；安装脚本会修改系统，先由服务器管理员审核：

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
sudo systemctl enable --now redis-server
ollama pull qwen2.5:7b
ollama list
redis-cli ping
```

`redis-cli ping` 应返回 `PONG`。代码连接本机 `localhost:6379`；Redis 断开时会退化为进程内存，重启后会话不保留。Ollama 默认使用本机 `11434`，不应向公网开放。

### 4. 配置并首次建库

```bash
cp .env.example .env
nano .env
```

至少核对以下项（均相对于仓库根目录；使用绝对路径也可以）：

```dotenv
OLLAMA_MODEL=qwen2.5:7b
KNOWLEDGE_FILE=./knowledge_data
CHROMA_PATH=./chroma_db
EMBEDDING_MODEL_PATH=./models/Qwen3-8B-native
EMBEDDING_DEVICE=cuda
RERANKER_MODEL_PATH=./models/bge-reranker-v2-m3
RERANKER_ENABLED=true
RAGAS_DO_NOT_TRACK=true
```

切换 Embedding 模型或改变其向量规则时，要用原始文档重新建 Chroma；不要混用旧索引。首次没有 `chroma_db/` 时，服务启动会读取 `KNOWLEDGE_FILE` 的 TXT/XLSX 自动建库，知识量大时可能耗时较长。确保运行账号对项目目录有读写权限。`.env` 不得提交到仓库。

### 5. 启动与验收

前四步只需部署时做一次。以后手动启动时，在**服务器终端 A**按顺序运行以下命令；先确认 Ollama、Redis 正常，再启动 EnBot：

```bash
cd ~/enbot
sudo systemctl start ollama
sudo systemctl start redis-server
redis-cli ping
ollama list
source .venv/bin/activate
python -X utf8 main.py
```

`redis-cli ping` 应显示 `PONG`；`ollama list` 应列出 `qwen2.5:7b`。首次启动会加载本地模型；如果没有现成的 `chroma_db/`，还会用知识库文件建索引，需等待直到终端显示服务监听 `http://127.0.0.1:8000`。**不要关闭终端 A**，关闭会停止手动启动的服务。

另开一个**服务器终端 B**，执行下面两条检查命令：

```bash
curl -fsS http://127.0.0.1:8000/
curl -fsS -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"HESS-HY-T-4~20K-M逆变器显示的电网侧数据不对","evaluate":false,"debug":false}'
```

第一条应包含 `"status":"ok"`，第二条应包含 `"status":"success"`，并给出与知识库相符的答案。**第一条成功不等于知识库正常**：如果第二条失败或只给出通用回答，回到终端 A 查看报错，检查模型目录、`.env` 中的知识库路径和 `chroma_db/` 是否为空。可再运行 `nvidia-smi` 查看显存占用。

如需从**自己的电脑**使用 Postman，而不是在服务器上执行 `curl`，先在自己电脑的终端建立 SSH 隧道（替换账号和 IP，保持这个终端打开）：

```bash
ssh -L 18000:127.0.0.1:8000 <登录账号>@<服务器IP>
```

随后在自己电脑浏览器打开 `http://127.0.0.1:18000/docs`；Postman 选择 **POST**，地址填 `http://127.0.0.1:18000/chat`，Body 选择 **raw → JSON**，粘贴上面 `-d` 后面的 JSON。现有 `/chat` 没有身份认证，且会话 ID 固定为 `default`；**未加鉴权、用户隔离和 HTTPS 前，不能直接公开 8000 端口或作为多用户公网服务**。

### 6. 验收后设置开机自启

确认第 5 步问答测试通过后，回到**终端 A 按 `Ctrl+C` 停止手动服务**，避免占用 8000 端口。然后在服务器终端执行 `sudo nano /etc/systemd/system/enbot.service`，粘贴以下内容。把三处 `<登录账号>` 替换成服务器实际登录账号；如果仓库不在该账号的 `/home/...` 目录，也要改成真实绝对路径。粘贴后按 `Ctrl+O`、回车、`Ctrl+X` 保存退出。

```ini
[Unit]
Description=EnBot API
After=network-online.target ollama.service redis-server.service
Wants=network-online.target

[Service]
Type=simple
User=<登录账号>
WorkingDirectory=/home/<登录账号>/enbot
ExecStart=/home/<登录账号>/enbot/.venv/bin/python -X utf8 main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

保存后在服务器执行：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now enbot
sudo systemctl status enbot --no-pager
sudo journalctl -u enbot -n 100 --no-pager
```

看到 `active (running)` 后，再执行第 5 步的两条 `curl` 命令；若失败，用 `sudo journalctl -u enbot -n 100 --no-pager` 查看原因。以后服务随服务器开机自动启动；手动重启用 `sudo systemctl restart enbot`。该项目尚无经压测的并发、认证和监控方案；对外提供服务前需另行补齐。
