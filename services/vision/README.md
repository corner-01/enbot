# EnBot 多模态视觉服务

该目录承载 Telebot 的图片理解入口，并与现有文本 RAG 隔离运行。

第一阶段处理链路：

1. 校验并预处理用户图片；
2. 使用本地 OCR 提取型号、故障码、数值、单位和页面文字；
3. 使用本地视觉模型理解屏幕、接线关系和安装场景；
4. 对 OCR 与视觉模型结果做确定性融合和冲突检查；
5. 将结构化结果转换为文本问题，调用现有 Milvus RAG；
6. 返回有知识库依据的诊断结果。

## 服务边界

- 独立虚拟环境：`D:\telebot\services\vision\.venv`
- 计划端口：`8011`
- 现有文本 API：继续使用 `8000`
- OCR：CPU 执行
- 视觉模型：通过本地 Ollama 调用，GPU 并发限制为 1
- 当前 Milvus 文本集合不存储图片向量

## 文件职责

- `app.py`：视觉服务 HTTP 入口与健康检查
- `schemas.py`：请求、识别结果和诊断结果的数据结构
- `image_validator.py`：格式、大小、清晰度和安全校验
- `image_preprocessor.py`：缩放、旋转、增强等预处理
- `ocr_engine.py`：PP-OCRv5 Mobile 本地 OCR 适配层，负责懒加载、置信度过滤、坐标保留和数值单位结构化
- `vision_client.py`：Ollama 视觉模型适配层
- `result_fusion.py`：OCR 与视觉模型结果融合和冲突检测
- `diagnosis_pipeline.py`：组织图片识别、文本 RAG 和最终回答
- `prompts.py`：视觉提取提示词

## OCR 当前状态

- 运行设备：CPU，不占用大模型所需显存
- 检测模型：`PP-OCRv5_mobile_det`
- 识别模型：`PP-OCRv5_mobile_rec`
- 模型缓存：`D:\AI-Cache\paddle`
- 默认最低置信度：`0.5`
- 模型按需加载，并在后续请求中复用
- 已支持从 OCR 文本中提取并规范化 A、V、W、Wh、Hz、百分比和温度等工程量

安装依赖：

```powershell
D:\telebot\services\vision\.venv\Scripts\python.exe -m pip install -r D:\telebot\services\vision\requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

运行 OCR 基准：

```powershell
$env:PADDLE_PDX_CACHE_HOME='D:\AI-Cache\paddle'
$env:PADDLE_PDX_MODEL_SOURCE='BOS'
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True'
D:\telebot\services\vision\.venv\Scripts\python.exe -X utf8 D:\telebot\evaluation\multimodal\benchmark_ocr.py --runs 3
```

基准结果保存在 `D:\telebot\evaluation\multimodal\results\mobile-warm-benchmark.json`。该结果目录默认不提交到 Git。

## 图片校验与预处理状态

- 校验图片完整性、格式、文件大小、尺寸和总像素量
- 计算清晰度、亮度和对比度，质量问题作为警告返回
- 根据 EXIF 修正手机照片方向
- 最长边超过 2048 像素时等比例缩小，不放大小图
- 分别生成保留颜色的 VLM 图片和灰度 OCR 图片
- 仅在低对比度或低清晰度时启用 CLAHE/轻微锐化
- 预处理文件保存在 `D:\telebot\tmp\vision-preprocessed`

## 本地视觉模型状态

- 模型：Ollama `qwen3-vl:2b-instruct`
- 接口：`POST http://127.0.0.1:11434/api/chat`
- 输入：预处理后的 VLM 彩色 PNG，以 Base64 发送
- 输出：使用 JSON Schema 强制约束，并用 Pydantic 再次验证
- 推理参数：`temperature=0`、`seed=42`、默认最多 1600 个输出 Token
- 并发：默认限制为 1，避免单 GPU 并发抢占显存
- 安全规则：明确的非 EnValt/Hanchu 厂商强制判为域外；无显式告警或故障码证据的故障项会被移除
- 性能元数据：记录加载耗时、总耗时、输入/输出 Token 和输出速度

## OCR/VLM 融合状态

- 数值与单位只采用 OCR 结构化结果
- VLM 厂商和产品型号必须能在 OCR 文本中找到，否则记录冲突并移除
- OCR 中明确出现 EnValt/Hanchu 时允许进入知识库检索
- 明确的外部厂商图片设置 `retrieval_allowed=false`，检索查询置空
- VLM 故障必须有 OCR 故障码、告警或错误文字支持
- 连接关系保留用于审计，但 OCR 只能证明组件文字出现，不能证明连接方向
- 冲突、低置信度丢弃项和未验证连接统一写入警告列表

## 诊断编排状态

- 自动执行预处理、OCR、Qwen3-VL、融合和 EnBot RAG
- OCR 与 Qwen3-VL 并行执行，减少端到端等待时间
- 域内图片通过 `POST /v1/chat` 调用现有 8000 端口服务
- 域外图片不调用 EnBot API，直接返回安全拒绝结果
- 返回各阶段耗时、知识库来源、请求编号和融合证据

命令行验证：

```powershell
D:\telebot\services\vision\.venv\Scripts\python.exe -X utf8 `
  D:\telebot\evaluation\multimodal\run_diagnosis_pipeline.py `
  --image "D:\path\to\image.png" `
  --question "What fault is shown?" `
  --session-id "vision-demo"
```

添加 `--debug` 可输出预处理、OCR 和 VLM 原始结构；添加 `--output D:\result.json` 可保存结果。

OCR、图片校验、图片预处理、视觉模型客户端、结果融合和诊断编排已经完成；下一步接入独立 HTTP 接口。
