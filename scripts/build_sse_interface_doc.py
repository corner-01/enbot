from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_ROW_HEIGHT_RULE, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "docs"
OUT_FILE = OUT_DIR / "EnBot_SSE接口对接说明.docx"

FONT_CN = "Microsoft YaHei"
FONT_CODE = "Consolas"
NAVY = "1F4E78"
PALE_BLUE = "EAF2F8"
PALE_GRAY = "F5F7F9"
BORDER = "D9E2F3"
TEXT = "1F1F1F"
MUTED = "5B6573"


def set_run_font(run, name=FONT_CN, size=10.5, bold=False, color=TEXT):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_border(cell, color=BORDER, size="6"):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        element = borders.find(tag)
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:color"), color)


def set_cell_margins(cell, top=110, start=130, bottom=110, end=130):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for name, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def prevent_row_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def add_body(doc, text, bold_lead=None):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.line_spacing = 1.35
    if bold_lead and text.startswith(bold_lead):
        lead = paragraph.add_run(bold_lead)
        set_run_font(lead, bold=True)
        rest = paragraph.add_run(text[len(bold_lead):])
        set_run_font(rest)
    else:
        run = paragraph.add_run(text)
        set_run_font(run)
    return paragraph


def add_bullets(doc, items):
    for item in items:
        paragraph = doc.add_paragraph(style="List Bullet")
        paragraph.paragraph_format.left_indent = Cm(0.65)
        paragraph.paragraph_format.first_line_indent = Cm(-0.25)
        paragraph.paragraph_format.space_after = Pt(3)
        paragraph.paragraph_format.line_spacing = 1.25
        set_run_font(paragraph.add_run(item))


def add_numbered(doc, items):
    for item in items:
        paragraph = doc.add_paragraph(style="List Number")
        paragraph.paragraph_format.left_indent = Cm(0.72)
        paragraph.paragraph_format.first_line_indent = Cm(-0.28)
        paragraph.paragraph_format.space_after = Pt(3)
        paragraph.paragraph_format.line_spacing = 1.25
        set_run_font(paragraph.add_run(item))


def add_code(doc, code):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.left_indent = Cm(0.45)
    paragraph.paragraph_format.right_indent = Cm(0.45)
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(7)
    paragraph.paragraph_format.line_spacing = 1.0
    p_pr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), PALE_GRAY)
    p_pr.append(shd)
    spacing = OxmlElement("w:spacing")
    spacing.set(qn("w:before"), "90")
    spacing.set(qn("w:after"), "90")
    p_pr.append(spacing)
    lines = code.strip("\n").splitlines()
    for index, line in enumerate(lines):
        run = paragraph.add_run(line)
        set_run_font(run, name=FONT_CODE, size=8.8, color="202124")
        if index < len(lines) - 1:
            run.add_break()
    return paragraph


def add_table(doc, headers, rows, widths=None, font_size=9.2):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    header = table.rows[0]
    repeat_header(header)
    prevent_row_split(header)
    for index, text in enumerate(headers):
        cell = header.cells[index]
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_shading(cell, NAVY)
        set_cell_border(cell)
        set_cell_margins(cell)
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(0)
        set_run_font(paragraph.add_run(text), size=9.2, bold=True, color="FFFFFF")
        if widths:
            cell.width = Cm(widths[index])
    for row_index, values in enumerate(rows):
        row = table.add_row()
        prevent_row_split(row)
        row.height_rule = WD_ROW_HEIGHT_RULE.AT_LEAST
        if row_index % 2:
            for cell in row.cells:
                set_cell_shading(cell, "F8FBFD")
        for index, value in enumerate(values):
            cell = row.cells[index]
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_border(cell)
            set_cell_margins(cell)
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.15
            if index == 0 and len(headers) <= 3:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_run_font(paragraph.add_run(str(value)), size=font_size)
            if widths:
                cell.width = Cm(widths[index])
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def configure_styles(doc):
    normal = doc.styles["Normal"]
    normal.font.name = FONT_CN
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_CN)
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor.from_string(TEXT)

    title = doc.styles["Title"]
    title.font.name = FONT_CN
    title._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_CN)
    title.font.size = Pt(22)
    title.font.bold = True
    title.font.color.rgb = RGBColor.from_string("000000")

    for style_name, size in (("Heading 1", 15), ("Heading 2", 12)):
        style = doc.styles[style_name]
        style.font.name = FONT_CN
        style._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_CN)
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string("000000")
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(12 if style_name == "Heading 1" else 8)
        style.paragraph_format.space_after = Pt(5)


def add_heading(doc, text, level=1):
    paragraph = doc.add_heading(text, level=level)
    paragraph.paragraph_format.keep_with_next = True
    return paragraph


def build_document():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = Document()
    configure_styles(doc)
    section = doc.sections[0]
    section.top_margin = Cm(2.1)
    section.bottom_margin = Cm(1.8)
    section.left_margin = Cm(2.25)
    section.right_margin = Cm(2.25)

    props = doc.core_properties
    props.title = "EnBot SSE接口对接说明"
    props.subject = "EnBot后端SSE接口交付与联调说明"
    props.author = "EnBot项目组"
    props.keywords = "EnBot,SSE,API,接口"

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(22)
    title.paragraph_format.space_after = Pt(8)
    title.add_run("EnBot SSE接口对接说明")

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(18)
    set_run_font(subtitle.add_run("后端接口交付文档"), size=11.5, color=MUTED)

    add_table(
        doc,
        ["项目", "内容"],
        [
            ("文档版本", "1.0"),
            ("接口版本", "EnBot API 3.2.0"),
            ("编制日期", "2026年10月9日"),
            ("适用范围", "EnBot文本问答SSE接口联调"),
        ],
        widths=[4.0, 11.0],
        font_size=9.5,
    )

    add_heading(doc, "文档目的", 1)
    add_body(
        doc,
        "本文档用于交付EnBot后端SSE接口。调用方按照本文档提交问题、识别事件并拼接回答即可完成联调。本文档不涉及模型训练、知识库内部实现或页面设计。",
    )
    add_body(
        doc,
        "交付结论：后端已提供POST /v1/chat/stream接口，并支持meta、heartbeat、delta、sources、done和error事件。原POST /v1/chat接口继续保留。",
        bold_lead="交付结论：",
    )

    add_heading(doc, "1 接口速查", 1)
    add_table(
        doc,
        ["项目", "约定"],
        [
            ("请求方法", "POST"),
            ("接口路径", "/v1/chat/stream"),
            ("响应类型", "text/event-stream; charset=utf-8"),
            ("本机地址", "http://127.0.0.1:8000/v1/chat/stream"),
            ("局域网联调地址", "http://192.168.3.165:8000/v1/chat/stream（开放局域网监听后使用）"),
            ("鉴权", "设置ENBOT_API_KEY时，请求头必须携带X-API-Key"),
            ("建议超时", "至少180秒；连接期间可能收到heartbeat事件"),
        ],
        widths=[4.1, 10.9],
        font_size=9.2,
    )

    doc.add_page_break()
    add_heading(doc, "2 请求规范", 1)
    add_heading(doc, "2.1 请求头", 2)
    add_code(
        doc,
        """Content-Type: application/json
Accept: text/event-stream
X-API-Key: <仅在后端启用鉴权时提供>
X-Request-ID: <可选的调用链路编号>""",
    )

    add_heading(doc, "2.2 请求体", 2)
    add_code(
        doc,
        """{
  "query": "hanchu的逆变器能不能光伏超配？",
  "session_id": "user-001",
  "evaluate": false,
  "debug": false
}""",
    )
    add_table(
        doc,
        ["字段", "类型", "是否必填", "说明"],
        [
            ("query", "string", "是", "用户问题；去除首尾空格后长度为1至8000。"),
            ("session_id", "string", "否", "会话编号，默认default；同一会话保持一致，不同用户不要共用。"),
            ("evaluate", "boolean", "否", "是否执行在线评测，默认false；生产环境通常关闭。"),
            ("ground_truth", "string或null", "否", "仅在启用在线评测时提供参考答案。"),
            ("debug", "boolean", "否", "是否返回检索调试信息，默认false；生产环境通常关闭。"),
        ],
        widths=[2.5, 2.6, 2.3, 7.8],
        font_size=8.8,
    )
    add_body(
        doc,
        "session_id只能包含英文字母、数字以及_.:@-，最大长度为128个字符。建议使用不包含手机号、邮箱等个人信息的内部会话编号。",
    )

    add_heading(doc, "3 SSE响应协议", 1)
    add_body(doc, "正常事件顺序如下。heartbeat可能不出现，也可能在处理时间较长时出现多次。")
    sequence = doc.add_paragraph()
    sequence.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sequence.paragraph_format.space_before = Pt(4)
    sequence.paragraph_format.space_after = Pt(8)
    set_run_font(
        sequence.add_run("meta  →  heartbeat  →  delta  →  sources  →  done"),
        name=FONT_CODE,
        size=10.2,
        bold=True,
        color=NAVY,
    )
    add_body(doc, "失败事件顺序通常为meta → error。error为终止事件，收到后不应继续等待done。")
    add_table(
        doc,
        ["事件", "出现次数", "调用方处理要求"],
        [
            ("meta", "1次", "保存request_id、session_id、model和protocol_version，确认连接已建立。"),
            ("heartbeat", "0至多次", "表示检索或生成仍在进行；保持连接，不作为回答文本显示。"),
            ("delta", "1至多次", "按照index顺序，将delta字段追加到当前回答，不能覆盖前文。"),
            ("sources", "1次", "读取sources和retrieval_backend，替换当前知识来源列表。"),
            ("done", "1次", "成功终止事件；结束加载状态并关闭本次读取。"),
            ("error", "失败时1次", "失败终止事件；显示message并结束本次读取。"),
        ],
        widths=[2.4, 2.4, 10.5],
        font_size=8.9,
    )

    doc.add_page_break()
    add_heading(doc, "4 事件示例", 1)
    add_heading(doc, "4.1 元数据事件", 2)
    add_code(
        doc,
        """id: abc123:0
event: meta
data: {"request_id":"abc123","session_id":"user-001","model":"enbot-v9:7b","protocol_version":"1.0"}""",
    )
    add_heading(doc, "4.2 心跳事件", 2)
    add_code(
        doc,
        """id: abc123:1
event: heartbeat
data: {"request_id":"abc123","elapsed_seconds":10.002}""",
    )
    add_heading(doc, "4.3 回答分片事件", 2)
    add_code(
        doc,
        """id: abc123:2
event: delta
data: {"request_id":"abc123","index":0,"delta":"支持1.5倍的光伏"}

id: abc123:3
event: delta
data: {"request_id":"abc123","index":1,"delta":"超配，但需要注意……"}""",
    )
    add_heading(doc, "4.4 来源与完成事件", 2)
    add_code(
        doc,
        """id: abc123:4
event: sources
data: {"request_id":"abc123","sources":["售前\\用户场景答疑.xlsx（第6行）"],"retrieval_backend":"milvus"}

id: abc123:5
event: done
data: {"request_id":"abc123","status":"success","finish_reason":"stop","duration_seconds":2.806}""",
    )
    add_heading(doc, "4.5 错误事件", 2)
    add_code(
        doc,
        """id: abc123:2
event: error
data: {"request_id":"abc123","code":"internal_error","status_code":503,"message":"服务暂时不可用"}""",
    )

    add_heading(doc, "5 调用方处理要求", 1)
    add_numbered(
        doc,
        [
            "以空行作为一个SSE事件帧的结束标记，读取event字段并将所有data行合并后解析JSON。",
            "收到delta后按index顺序追加文本；不要用新分片覆盖已经显示的回答。",
            "收到heartbeat只更新等待状态，不把心跳数据拼接到回答中。",
            "收到sources后展示或保存知识来源；sources为空不表示请求失败。",
            "收到done或error后结束读取；若连接断开且没有终止事件，应按异常处理。",
            "调用方主动停止生成时可以关闭HTTP连接。服务端会检测连接断开并清理运行状态。",
        ],
    )
    add_body(
        doc,
        "重要说明：当前接口会立即建立SSE连接，并在RAG检索和模型生成期间发送heartbeat；完整回答生成完成后，再通过多个delta事件分片发送。当前不是模型逐Token实时输出。",
        bold_lead="重要说明：",
    )

    doc.add_page_break()
    add_heading(doc, "6 错误处理", 1)
    add_heading(doc, "6.1 建立SSE连接前的HTTP错误", 2)
    add_table(
        doc,
        ["HTTP状态码", "含义", "建议处理"],
        [
            ("401", "API Key缺失或错误", "检查X-API-Key；密钥应通过安全渠道单独传递。"),
            ("403", "请求使用了未启用的evaluate或debug功能", "将相应字段设置为false。"),
            ("422", "请求字段、类型或session_id格式不合法", "根据返回的字段错误修改请求。"),
            ("503", "模型、Milvus或运行时尚未就绪", "检查/health/ready，稍后重试。"),
        ],
        widths=[2.7, 5.3, 7.3],
        font_size=9.0,
    )
    add_heading(doc, "6.2 连接建立后的流内错误", 2)
    add_body(doc, "SSE连接建立后发生异常时，服务端发送error事件并结束事件流。调用方应优先展示error.data.message，同时保留request_id用于定位日志。")

    add_heading(doc, "7 联调命令", 1)
    add_body(doc, "在后端电脑上可以使用以下PowerShell命令验证事件流：")
    add_code(
        doc,
        """curl.exe -N --max-time 180 `
  -H "Content-Type: application/json" `
  -H "Accept: text/event-stream" `
  --data-raw '{"query":"hanchu的逆变器能不能光伏超配？","session_id":"sse-demo","evaluate":false,"debug":false}' `
  http://127.0.0.1:8000/v1/chat/stream""",
    )
    add_body(doc, "预期至少依次看到meta、一个或多个delta、sources和done。处理时间超过心跳间隔时，还会看到heartbeat。")

    add_heading(doc, "8 局域网联调前提", 1)
    add_bullets(
        doc,
        [
            "后端服务必须监听0.0.0.0:8000，而不是仅监听127.0.0.1。",
            "前后端电脑必须能够通过局域网互相访问；当前后端电脑局域网地址为192.168.3.165。",
            "Windows防火墙只在专用网络中开放TCP 8000，并尽量限制允许访问的来源IP。",
            "浏览器跨域直连时，后端需要把前端页面的完整Origin加入ENBOT_CORS_ORIGINS。",
            "API Key不得写入公开文档、截图或代码仓库，应通过安全渠道单独提供。",
            "经过Nginx等反向代理时必须关闭响应缓冲，否则SSE事件可能被合并后一次性返回。",
        ],
    )

    add_heading(doc, "9 联调验收清单", 1)
    add_table(
        doc,
        ["检查项", "验收标准"],
        [
            ("服务状态", "/health/ready返回status=ready，Milvus和enbot-v9:7b均可用。"),
            ("连接建立", "POST请求返回200和text/event-stream。"),
            ("元数据", "首个业务事件为meta，并包含request_id。"),
            ("回答拼接", "多个delta按index顺序拼接后得到完整中文回答。"),
            ("知识来源", "sources事件能返回来源列表和milvus检索后端。"),
            ("成功结束", "正常请求最终收到done。"),
            ("异常结束", "服务异常时收到error或明确的HTTP错误。"),
            ("长连接", "耗时请求期间heartbeat能够保持连接。"),
            ("会话隔离", "不同用户使用不同session_id，同一对话保持相同session_id。"),
        ],
        widths=[4.0, 11.2],
        font_size=9.0,
    )

    doc.add_page_break()
    add_heading(doc, "附 给前端的联调说明原文", 1)
    add_code(
        doc,
        """> 前端同学，后端 SSE 接口已经完成，请按下面协议进行联调。
>
> 接口地址：
> `POST http://192.168.3.165:8000/v1/chat/stream`
>
> 请求头：
> `Content-Type: application/json`
> `Accept: text/event-stream`
>
> 请求体：
> ```json
> {
>   "query": "用户问题",
>   "session_id": "user-001",
>   "evaluate": false,
>   "debug": false
> }
> ```
>
> 返回事件顺序一般为：
> `meta → heartbeat → delta → sources → done`
>
> - `meta`：返回 `request_id`、会话编号和模型信息
> - `heartbeat`：后端还在检索或生成，收到后继续等待
> - `delta`：回答分片，按照 `index` 顺序追加显示
> - `sources`：知识库来源和检索后端
> - `done`：回答正常结束
> - `error`：发生异常时返回，收到后结束本次请求
>
> 请注意：这是 POST 请求，不能直接使用原生 `EventSource`，需要使用 `fetch` 读取响应流。多个 `delta` 需要拼接，不能覆盖之前的内容。
>
> 当前接口已经通过后端测试。联调时请把前端页面的完整地址和端口发给我，我配置跨域和局域网访问。
>
> 另外提醒一句：`192.168.3.165:8000` 要等我把后端改成局域网监听后，另一台电脑才能访问。""",
    )

    footer = section.footer
    footer_paragraph = footer.paragraphs[0]
    footer_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer_paragraph.paragraph_format.space_before = Pt(4)
    set_run_font(
        footer_paragraph.add_run("EnBot SSE接口对接说明  |  后端接口版本 3.2.0"),
        size=8.5,
        color=MUTED,
    )

    doc.save(OUT_FILE)
    print(OUT_FILE)


if __name__ == "__main__":
    build_document()
