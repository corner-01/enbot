from __future__ import annotations

from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "output" / "docs" / "EnBot双机GPU算力均摊部署操作手册.docx"
DARK_BLUE = "17365D"
PALE_BLUE = "F3F7FB"
LIGHT_GRAY = "D9D9D9"
CODE_GRAY = "F3F4F6"


def set_run_font(run, name: str, size: float, bold: bool = False, color: str = "000000"):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.get_or_add_rFonts()
    fonts.set(qn("w:ascii"), name)
    fonts.set(qn("w:hAnsi"), name)
    fonts.set(qn("w:eastAsia"), name)


def set_cell_shading(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = tc_pr.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        tc_pr.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=100, start=120, bottom=100, end=120):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color: str = LIGHT_GRAY):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "6")
        element.set(qn("w:color"), color)


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    marker = OxmlElement("w:tblHeader")
    marker.set(qn("w:val"), "true")
    tr_pr.append(marker)


def set_paragraph_shading(paragraph, fill: str):
    p_pr = paragraph._p.get_or_add_pPr()
    shading = p_pr.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        p_pr.append(shading)
    shading.set(qn("w:fill"), fill)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_separate = OxmlElement("w:fldChar")
    fld_separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_begin, instr, fld_separate, text, fld_end])
    set_run_font(run, "Arial", 9, color="666666")


def add_body(doc: Document, text: str, bold_lead: str | None = None):
    paragraph = doc.add_paragraph()
    if bold_lead and text.startswith(bold_lead):
        first = paragraph.add_run(bold_lead)
        set_run_font(first, "SimSun", 11, bold=True)
        rest = paragraph.add_run(text[len(bold_lead):])
        set_run_font(rest, "SimSun", 11)
    else:
        set_run_font(paragraph.add_run(text), "SimSun", 11)
    paragraph.paragraph_format.first_line_indent = Inches(0.28)
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.line_spacing = 1.25
    return paragraph


def add_bullet(doc: Document, text: str):
    paragraph = doc.add_paragraph(style="List Bullet")
    set_run_font(paragraph.add_run(text), "SimSun", 11)
    paragraph.paragraph_format.space_after = Pt(3)
    return paragraph


def add_numbered(doc: Document, text: str):
    paragraph = doc.add_paragraph(style="List Number")
    set_run_font(paragraph.add_run(text), "SimSun", 11)
    paragraph.paragraph_format.space_after = Pt(3)
    return paragraph


def add_code(doc: Document, code: str):
    paragraph = doc.add_paragraph()
    set_paragraph_shading(paragraph, CODE_GRAY)
    paragraph.paragraph_format.left_indent = Inches(0.18)
    paragraph.paragraph_format.right_indent = Inches(0.18)
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(7)
    paragraph.paragraph_format.line_spacing = 1.05
    set_run_font(paragraph.add_run(code.strip()), "Consolas", 9.5)
    return paragraph


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float]):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    header = table.rows[0]
    set_repeat_table_header(header)
    for index, (cell, value) in enumerate(zip(header.cells, headers)):
        cell.width = Inches(widths[index])
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_shading(cell, DARK_BLUE)
        set_cell_margins(cell)
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_run_font(paragraph.add_run(value), "Microsoft YaHei", 10, bold=True, color="FFFFFF")
    for row_index, values in enumerate(rows):
        row = table.add_row()
        for index, (cell, value) in enumerate(zip(row.cells, values)):
            cell.width = Inches(widths[index])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell)
            if row_index % 2:
                set_cell_shading(cell, PALE_BLUE)
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if index == 0 else WD_ALIGN_PARAGRAPH.LEFT
            set_run_font(paragraph.add_run(value), "SimSun", 10)
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.1
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_heading(doc: Document, text: str, level: int):
    paragraph = doc.add_heading(text, level=level)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    paragraph.paragraph_format.space_after = Pt(5)
    for run in paragraph.runs:
        set_run_font(run, "Microsoft YaHei", 15 if level == 1 else 12, bold=True)
    return paragraph


def add_checklist(doc: Document, items: list[str]):
    for item in items:
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.left_indent = Inches(0.18)
        paragraph.paragraph_format.space_after = Pt(3)
        set_run_font(paragraph.add_run(f"[ ] {item}"), "SimSun", 10.5)


def configure_styles(doc: Document):
    normal = doc.styles["Normal"]
    normal.font.name = "SimSun"
    normal.font.size = Pt(11)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun")
    for style_name, size in (("Title", 24), ("Subtitle", 13)):
        style = doc.styles[style_name]
        style.font.name = "Microsoft YaHei"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")


def build_document():
    doc = Document()
    configure_styles(doc)
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.78)
    section.bottom_margin = Inches(0.72)
    section.left_margin = Inches(0.78)
    section.right_margin = Inches(0.78)
    add_page_number(section.footer.paragraphs[0])

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(115)
    title.paragraph_format.space_after = Pt(18)
    set_run_font(title.add_run("EnBot双机GPU算力均摊部署操作手册"), "Microsoft YaHei", 24, bold=True)

    subtitle = doc.add_paragraph(style="Subtitle")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(35)
    set_run_font(subtitle.add_run("主机A负责生成与应用  主机B负责检索与GPU重排"), "Microsoft YaHei", 13)

    add_table(doc, ["项目", "内容"], [
        ["适用项目", r"D:\telebot"],
        ["主机A", "当前RTX 4070电脑，运行主服务、Ollama文本生成、视觉模型和飞书"],
        ["主机B", "另一台NVIDIA GPU电脑，运行Embedding、Milvus、BM25、RRF和BGE Reranker"],
        ["网络", "两台电脑位于同一可信局域网，主机A通过HTTP调用主机B"],
        ["手册版本", date.today().isoformat()],
    ], [1.35, 5.25])
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(22)
    set_run_font(p.add_run("完成本手册后，飞书入口保持不变。"), "SimSun", 11, bold=True)

    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    add_heading(doc, "一 部署结果", 1)
    add_body(doc, "本方案将原来集中在一台电脑上的生成与检索负载拆成两个独立服务。主机A不再加载Qwen3-8B Embedding和BGE重排模型，释放显存给enbot-optuna 7B文本生成模型和qwen3-vl视觉模型。主机B集中完成知识库检索和GPU重排，只把最终文档返回主机A。")
    add_table(doc, ["节点", "主要服务", "GPU负载", "监听端口"], [
        ["主机A", "EnBot API、Ollama、视觉API、飞书、Redis", "7B生成模型和视觉模型", "8000、8011"],
        ["主机B", "Embedding、Milvus、BM25、RRF、BGE重排、检索API", "Qwen3-8B Embedding和BGE Reranker", "8100"],
    ], [0.85, 2.35, 2.3, 1.1])
    add_body(doc, "请求链路：用户提出问题后，主机A把问题发送给主机B。主机B依次执行问题向量化、Milvus向量检索、BM25关键词检索、RRF融合和GPU重排，返回最终两条文档。主机A再调用本地Ollama生成回答。", bold_lead="请求链路：")

    add_heading(doc, "二 部署前检查", 1)
    add_heading(doc, "2.1 主机B硬件和软件要求", 2)
    add_table(doc, ["检查项", "建议要求"], [
        ["显卡", "NVIDIA GPU，建议显存16GB或以上"],
        ["内存", "建议32GB或以上"],
        ["操作系统", "Windows 10或Windows 11 64位"],
        ["软件", "NVIDIA驱动、Python 3.10、Docker Desktop、Git"],
        ["网络", "与主机A在同一局域网，能够互相Ping通"],
        ["磁盘", "建议至少保留60GB可用空间"],
    ], [1.35, 5.25])
    add_body(doc, "在主机B打开PowerShell，依次执行以下命令并确认显卡与IPv4地址。")
    add_code(doc, "nvidia-smi\nipconfig")
    add_checklist(doc, ["nvidia-smi可以看到NVIDIA显卡名称和显存", "记下主机B的IPv4地址，例如192.168.1.120", "Docker Desktop可以正常启动", "主机A能够Ping通主机B"])

    add_heading(doc, "2.2 生成两台机器共用的密钥", 2)
    add_body(doc, "在任意一台电脑的PowerShell执行下面命令，复制输出结果。主机A和主机B必须使用完全相同的密钥。不要把密钥提交到GitHub。")
    add_code(doc, "[Convert]::ToHexString([Security.Cryptography.RandomNumberGenerator]::GetBytes(32)).ToLower()")

    add_heading(doc, "2.3 通过局域网传输模型和知识库", 2)
    add_body(doc, "代码通过Git仓库获取；模型和知识库体积较大，通过同一局域网直接复制。不要把模型权重提交到普通Git仓库，也不要复制venv目录。以下示例假设主机A的IPv4地址为192.168.1.100，共享名称为telebot-share。实际操作时必须替换为主机A的真实地址。")

    add_heading(doc, "2.3.1 主机A检查待传输内容", 3)
    add_body(doc, "在主机A打开PowerShell，确认两个模型目录和原始知识库目录都存在。Qwen3-8B模型约15.15GB，BGE重排模型约2.29GB。knowledge_data中必须保存用于建库的TXT、XLSX或JSON原始文件。")
    add_code(doc, r'''Test-Path D:\telebot\models\Qwen3-8B-native
Test-Path D:\telebot\models\bge-reranker-v2-m3
Test-Path D:\telebot\knowledge_data

Get-ChildItem D:\telebot\models\Qwen3-8B-native -Recurse -File |
  Measure-Object Length -Sum
Get-ChildItem D:\telebot\models\bge-reranker-v2-m3 -Recurse -File |
  Measure-Object Length -Sum
Get-ChildItem D:\telebot\knowledge_data -Recurse -File |
  Measure-Object Length -Sum''')
    add_body(doc, "前三条命令必须全部返回True。若knowledge_data返回False，应先找回或整理原始知识文件，再继续传输。不能只创建空目录，也不能只复制正在运行的Milvus数据目录；BM25关键词检索和父记录恢复都需要原始知识文件。", bold_lead="停止条件：")

    add_heading(doc, "2.3.2 主机A开启只读共享", 3)
    add_numbered(doc, "右键D:\\telebot，依次选择属性、共享、高级共享。")
    add_numbered(doc, "勾选共享此文件夹，把共享名设置为telebot-share。")
    add_numbered(doc, "在权限中只授予读取权限；传输完成后立即关闭共享。")
    add_numbered(doc, "确认Windows网络类型为专用网络，并允许文件和打印机共享。")
    add_body(doc, "在主机A执行ipconfig，记录当前局域网IPv4地址。不要使用127.0.0.1，也不要把公网地址填入下面的命令。")
    add_code(doc, "ipconfig")

    add_heading(doc, "2.3.3 主机B检查共享连接", 3)
    add_body(doc, "在主机B打开PowerShell，先确认能够访问主机A的SMB共享端口。把192.168.1.100替换为主机A实际IPv4地址。")
    add_code(doc, "Test-NetConnection 192.168.1.100 -Port 445\nGet-ChildItem \\\\192.168.1.100\\telebot-share")
    add_body(doc, "TcpTestSucceeded必须为True，并且第二条命令能够列出主机A项目目录。如果系统要求登录，输入主机A的Windows账户和密码，不要把密码写入脚本。")

    add_heading(doc, "2.3.4 主机B复制模型和知识库", 3)
    add_body(doc, "主机B先通过Git克隆项目代码，再使用robocopy复制大文件。/Z支持断点续传，/J适合大文件，/MT:8使用8个复制线程。传输中断时可以重新执行同一条命令。")
    add_code(doc, r'''cd D:\
git clone -b codex/optuna-retrieval-v1 https://cnb.cool/envalt-tech/enbot telebot

robocopy "\\192.168.1.100\telebot-share\models\Qwen3-8B-native" `
  "D:\telebot\models\Qwen3-8B-native" /E /Z /J /MT:8 /R:3 /W:3

robocopy "\\192.168.1.100\telebot-share\models\bge-reranker-v2-m3" `
  "D:\telebot\models\bge-reranker-v2-m3" /E /Z /J /MT:8 /R:3 /W:3

robocopy "\\192.168.1.100\telebot-share\knowledge_data" `
  "D:\telebot\knowledge_data" /E /Z /MT:8 /R:3 /W:3''')
    add_body(doc, "robocopy退出码0至7通常表示复制成功或存在可接受的差异；退出码8及以上表示失败，应检查网络、共享权限和磁盘空间后重新执行。不要因为终端显示非零退出码就直接删除已复制文件。")

    add_heading(doc, "2.3.5 主机B校验传输结果", 3)
    add_code(doc, r'''Test-Path D:\telebot\models\Qwen3-8B-native
Test-Path D:\telebot\models\bge-reranker-v2-m3
Test-Path D:\telebot\knowledge_data

Get-ChildItem D:\telebot\models\Qwen3-8B-native -Recurse -File |
  Measure-Object Length -Sum
Get-ChildItem D:\telebot\models\bge-reranker-v2-m3 -Recurse -File |
  Measure-Object Length -Sum
Get-ChildItem D:\telebot\knowledge_data -Recurse -File |
  Measure-Object Length -Sum''')
    add_body(doc, "三条Test-Path必须全部返回True。主机B统计出的文件数量和总字节数应与主机A一致。确认无误后，在主机A关闭telebot-share共享，再继续第三章的环境安装和服务启动。")

    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    add_heading(doc, "三 准备主机B", 1)
    add_heading(doc, "3.1 复制项目和模型", 2)
    add_body(doc, "将项目复制到主机B，例如D:\\telebot。建议通过Git获取代码，再单独复制模型和知识库。不要直接复制venv目录，因为虚拟环境包含原电脑路径。")
    add_bullet(doc, r"必须存在 D:\telebot\models\Qwen3-8B-native")
    add_bullet(doc, r"必须存在 D:\telebot\models\bge-reranker-v2-m3")
    add_bullet(doc, r"必须存在 D:\telebot\knowledge_data")
    add_bullet(doc, r"必须存在 D:\telebot\scripts\14_start_retrieval_gpu.ps1")

    add_heading(doc, "3.2 创建Python环境", 2)
    add_code(doc, r"""cd D:\telebot
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt""")
    add_body(doc, "如果PowerShell提示禁止执行脚本，只对当前窗口临时放行，然后重新激活虚拟环境。")
    add_code(doc, "Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass")

    add_heading(doc, "3.3 检查CUDA是否被PyTorch识别", 2)
    add_code(doc, r""".\venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))" """)
    add_body(doc, "第一行必须显示True，第二行应显示主机B的显卡名称。如果显示False，先修复NVIDIA驱动或PyTorch CUDA环境，不要继续启动检索服务。")

    add_heading(doc, "3.4 首次启动主机B", 2)
    add_body(doc, "先启动Docker Desktop。回到D:\\telebot PowerShell，将下面的same-secret替换为第二节生成的真实密钥。")
    add_code(doc, r""".\scripts\14_start_retrieval_gpu.ps1 -ApiKey "same-secret" """)
    add_body(doc, "脚本会检查模型目录和CUDA，启动Milvus，然后使用一个进程在8100端口启动检索API。首次启动会加载Qwen3-8B Embedding、建立或读取Milvus索引，并执行一次预热检索，耗时通常比后续启动更长。")
    add_body(doc, "重要：不要把workers改成2或更大。每增加一个工作进程都会重复加载一套Embedding和重排模型，容易造成显存不足。", bold_lead="重要：")

    add_heading(doc, "3.5 放行Windows防火墙", 2)
    add_body(doc, "第一次启动时，如果Windows询问是否允许Python访问网络，只勾选专用网络。若没有弹窗，可以管理员身份执行下面命令，只开放检索API端口。")
    add_code(doc, 'New-NetFirewallRule -DisplayName "EnBot Retrieval 8100" -Direction Inbound -Protocol TCP -LocalPort 8100 -Action Allow -Profile Private')
    add_body(doc, "不需要向局域网开放Milvus的19530端口。Milvus只由主机B本机访问。")

    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    add_heading(doc, "四 验证主机B", 1)
    add_heading(doc, "4.1 本机健康检查", 2)
    add_code(doc, "Invoke-RestMethod http://127.0.0.1:8100/health/live\nInvoke-RestMethod http://127.0.0.1:8100/health/ready")
    add_body(doc, "ready响应中的available应为True，embedding_device和reranker_device都应为cuda，并显示device_name。")

    add_heading(doc, "4.2 本机检索测试", 2)
    add_code(doc, r"""$headers = @{ "X-API-Key" = "same-secret" }
$body = @{ query = "逆变器显示Fault怎么处理" } | ConvertTo-Json
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8100/v1/retrieve" `
  -Headers $headers `
  -ContentType "application/json" `
  -Body $body | ConvertTo-Json -Depth 8""")
    add_body(doc, "返回结果中应出现status为success、retrieval_backend为milvus，并包含documents数组。每条文档应有content和metadata。")

    add_heading(doc, "4.3 确认重排正在使用GPU", 2)
    add_code(doc, "nvidia-smi -l 1")
    add_body(doc, "保持监控窗口运行，再执行一次检索。Python进程应出现在GPU进程列表中，显存占用应明显增加。服务终端还会显示Reranker loaded on cuda。")

    add_heading(doc, "五 连接主机A", 1)
    add_heading(doc, "5.1 从主机A测试网络", 2)
    add_body(doc, "下面示例假设主机B地址为192.168.1.120。请替换为实际地址。")
    add_code(doc, "Test-NetConnection 192.168.1.120 -Port 8100\nInvoke-RestMethod http://192.168.1.120:8100/health/ready")
    add_body(doc, "TcpTestSucceeded必须为True。如果失败，检查两台机器是否处于同一网络、主机B防火墙是否放行8100，以及主机B服务是否仍在运行。")

    add_heading(doc, "5.2 启动当前电脑的主服务", 2)
    add_body(doc, "先确认当前电脑的Ollama已经运行并且ollama list中存在enbot-v9:7b。然后执行项目自带的分布式启动脚本。")
    add_code(doc, r"""cd D:\telebot
.\scripts\13_start_primary_distributed.ps1 `
  -RetrievalApiUrl "http://192.168.1.120:8100" `
  -RetrievalApiKey "same-secret" """)
    add_body(doc, "该脚本会先检查主机B，再检查本机Ollama，随后把RETRIEVAL_MODE设置为remote，并清空OLLAMA_NUM_GPU限制。主机A因此不会加载本地Embedding，Ollama可以自动使用RTX 4070。")

    add_heading(doc, "5.3 验证主机A", 2)
    add_code(doc, r"""Invoke-RestMethod http://127.0.0.1:8000/health/ready

$body = @{
  query = "逆变器能不能光伏超配"
  session_id = "distributed-test"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/v1/chat" `
  -ContentType "application/json" `
  -Body $body | ConvertTo-Json -Depth 8""")
    add_body(doc, "回答应正常返回知识库来源。此时在主机A运行nvidia-smi，应主要看到Ollama生成模型；主机B应看到Embedding和Reranker。")

    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    add_heading(doc, "六 恢复飞书和视觉服务", 1)
    add_body(doc, "分布式改造只改变知识检索位置，不改变用户入口。主机A的8000端口启动成功后，在新的PowerShell窗口继续运行原有脚本。")
    add_table(doc, ["功能", "主机A命令"], [
        ["视觉诊断API", r".\scripts\11_start_vision_api.ps1"],
        ["飞书机器人", r".\scripts\07_start_feishu_bot.ps1"],
    ], [1.55, 5.05])
    add_body(doc, "推荐顺序：先启动主机B检索服务，再启动主机A主服务，最后启动视觉和飞书。关闭时按相反顺序操作。")

    add_heading(doc, "七 每日启动顺序", 1)
    for item in [
        "主机B启动Docker Desktop。",
        r'主机B运行 .\scripts\14_start_retrieval_gpu.ps1 -ApiKey "same-secret"。',
        "主机B确认health ready返回ready。",
        "主机A启动Ollama。",
        "主机A运行13_start_primary_distributed.ps1并传入主机B地址和相同密钥。",
        "主机A启动视觉API和飞书机器人。",
        "分别查看两台机器的nvidia-smi，确认负载落在预期机器。",
    ]:
        add_numbered(doc, item)

    add_heading(doc, "八 验收清单", 1)
    add_checklist(doc, [
        "主机B的health live返回ok",
        "主机B的health ready返回ready并显示CUDA可用",
        "主机B检索接口返回知识文档",
        "主机B日志显示Reranker loaded on cuda",
        "主机B的nvidia-smi出现Python检索进程",
        "主机A的health ready返回ready",
        "主机A可以完成知识库问答并显示来源",
        "主机A没有加载Qwen3-8B Embedding",
        "主机A的Ollama使用RTX 4070生成回答",
        "飞书文字和图片消息正常",
    ])

    add_heading(doc, "九 常见问题", 1)
    add_table(doc, ["现象", "原因", "处理方法"], [
        ["主机B显示CUDA unavailable", "PyTorch未识别CUDA或驱动异常", "运行nvidia-smi和torch.cuda.is_available，重新安装匹配CUDA的PyTorch"],
        ["主机A连接8100失败", "防火墙、IP地址或服务未启动", "执行Test-NetConnection，检查主机B地址和防火墙专用网络规则"],
        ["接口返回401", "两台机器密钥不一致", "重新输入完全相同的RETRIEVAL_API_KEY"],
        ["主机B显存不足", "Embedding和Reranker同时加载超过显存", "将EMBEDDING_BATCH_SIZE降为1或2；仍不足时需更大显存或较小Embedding模型"],
        ["主机A仍占满Embedding显存", "没有通过13号脚本启动或RETRIEVAL_MODE不是remote", "停止服务后使用13_start_primary_distributed.ps1重新启动"],
        ["首次请求很慢", "模型与Milvus索引正在预热", "等待health ready完成后再测试；后续请求会更快"],
        ["Milvus连接失败", "Docker Desktop或Milvus容器未运行", "先启动Docker，再运行09_start_milvus.ps1"],
        ["回答没有知识依据", "主机B未返回文档或问题超出知识库", "直接调用8100的v1 retrieve检查documents内容"],
    ], [1.4, 2.1, 3.1])

    add_heading(doc, "十 文件说明", 1)
    add_table(doc, ["文件", "用途"], [
        [r"services\retrieval\app.py", "主机B独立检索API"],
        [r"rag\remote_retriever.py", "主机A远程检索客户端"],
        [r"rag\knowledge_retriever.py", "根据RETRIEVAL_MODE选择本地或远程检索"],
        [r"rag\hybrid_search.py", "支持RERANKER_DEVICE cuda的GPU重排"],
        [r"scripts\13_start_primary_distributed.ps1", "启动主机A"],
        [r"scripts\14_start_retrieval_gpu.ps1", "启动主机B"],
        [r"deployment\distributed\primary.env.example", "主机A配置样例"],
        [r"deployment\distributed\retrieval.env.example", "主机B配置样例"],
    ], [2.9, 3.7])

    add_heading(doc, "十一 安全要求", 1)
    for item in [
        "8100端口只允许可信局域网访问，不要直接映射到公网。",
        "主机A和主机B必须设置相同的长随机API密钥。",
        "不要把真实密钥写入Git仓库、截图或公开报告。",
        "Milvus的19530端口只在主机B本机使用。",
        "若跨地点部署，优先使用VPN或Tailscale，不要裸露HTTP服务。",
    ]:
        add_bullet(doc, item)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT_PATH)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    build_document()
