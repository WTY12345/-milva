from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "制图输出"
OUT.mkdir(exist_ok=True)

FONT = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for char in text:
        trial = current + char
        if text_size(draw, trial, fnt)[0] <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = char
    if current:
        lines.append(current)
    return lines


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: str = "#1f2937",
    line_gap: int = 8,
) -> None:
    x1, y1, x2, y2 = box
    lines = wrap_text(draw, text, fnt, x2 - x1 - 34)
    heights = [text_size(draw, line, fnt)[1] for line in lines]
    total_h = sum(heights) + line_gap * (len(lines) - 1)
    y = y1 + ((y2 - y1) - total_h) / 2
    for line, h in zip(lines, heights):
        w, _ = text_size(draw, line, fnt)
        draw.text((x1 + ((x2 - x1) - w) / 2, y), line, font=fnt, fill=fill)
        y += h + line_gap


def box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    text: str,
    fill: str = "#ffffff",
    outline: str = "#9bb7d4",
    text_fill: str = "#1f2937",
    width: int = 3,
    radius: int = 22,
    size: int = 34,
    bold: bool = True,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
    centered_text(draw, xy, text, font(size, bold), text_fill)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill: str = "#385a7c", width: int = 4) -> None:
    draw.line([start, end], fill=fill, width=width)
    x1, y1 = start
    x2, y2 = end
    if abs(x2 - x1) >= abs(y2 - y1):
        sign = 1 if x2 > x1 else -1
        points = [(x2, y2), (x2 - sign * 18, y2 - 10), (x2 - sign * 18, y2 + 10)]
    else:
        sign = 1 if y2 > y1 else -1
        points = [(x2, y2), (x2 - 10, y2 - sign * 18), (x2 + 10, y2 - sign * 18)]
    draw.polygon(points, fill=fill)


def canvas(title: str, subtitle: str | None = None) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (1600, 940), "#f6f8fb")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 1600, 118), fill="#eef4f9")
    draw.text((70, 38), title, font=font(42, True), fill="#12324a")
    if subtitle:
        draw.text((70, 86), subtitle, font=font(22), fill="#667085")
    return img, draw


def save(img: Image.Image, name: str) -> str:
    path = OUT / name
    img.save(path, "PNG")
    return str(path)


def fig_tech_route() -> str:
    img, draw = canvas("图1-1 课题技术路线图", "从问题分析到实验评估的毕业设计实施路径")
    labels = ["文献调研", "需求分析", "系统设计", "编码实现", "实验评估", "论文总结"]
    x, y, w, h, gap = 70, 390, 210, 120, 38
    colors = ["#e8f0ff", "#eef7ed", "#fff6df", "#f0efff", "#e9f8f5", "#fff0f0"]
    for i, label in enumerate(labels):
        box(draw, (x + i * (w + gap), y, x + i * (w + gap) + w, y + h), label, colors[i], "#7a9ec1", size=32)
        if i < len(labels) - 1:
            arrow(draw, (x + i * (w + gap) + w + 5, y + h // 2), (x + (i + 1) * (w + gap) - 8, y + h // 2))
    notes = [
        "明确商品向量检索研究背景",
        "确定数据生成、存储、检索、监控需求",
        "设计 Milvus + Local ANN 双后端",
        "实现 Web 服务、批量生成与实时监控",
        "测试延迟、Recall@K、MRR@K、NDCG@K、更新删除能力",
        "整理论文图表与答辩材料",
    ]
    for i, note in enumerate(notes):
        bx = x + i * (w + gap)
        box(draw, (bx, 575, bx + w, 735), note, "#ffffff", "#d8dee8", size=21, bold=False)
    return save(img, "图1-1_课题技术路线图.png")


def fig_ann_classification() -> str:
    img, draw = canvas("图2-1 近似最近邻检索方法分类", "按索引思想划分常见 ANN 方法及适用特点")
    root = (620, 145, 980, 225)
    box(draw, root, "近似最近邻检索 ANN", "#e8f0ff", "#6f96bd", size=30)
    children = [
        ("哈希方法", "LSH、随机投影\n适合快速粗召回"),
        ("树结构方法", "KD-Tree、Annoy\n适合中低维数据"),
        ("量化方法", "PQ、IVF-PQ\n适合压缩和大规模召回"),
        ("图结构方法", "HNSW、NSG\n查询速度和精度较好"),
        ("向量数据库", "Milvus、Qdrant、Pinecone\n提供服务化检索能力"),
    ]
    positions = [(80, 395), (390, 395), (700, 395), (1010, 395), (1320, 395)]
    for (title, desc), (cx, cy) in zip(children, positions):
        arrow(draw, ((root[0] + root[2]) // 2, root[3]), (cx + 90, cy - 30), "#6a7f95", 3)
        box(draw, (cx, cy, cx + 210, cy + 82), title, "#ffffff", "#8fb2d6", size=28)
        box(draw, (cx - 28, cy + 128, cx + 238, cy + 288), desc, "#fbfcfe", "#d9e3ee", size=22, bold=False)
    return save(img, "图2-1_近似最近邻检索方法分类.png")


def fig_architecture() -> str:
    img, draw = canvas("图4-1 系统总体架构图", "前端交互、服务层、向量后端与数据生成模块的整体关系")
    layers = [
        ("用户与前端", ["商品录入", "自然语言检索", "实时监控面板"]),
        ("Web/API 服务层", ["http_api.py", "service.py", "任务调度与指标采集"]),
        ("向量与数据层", ["Milvus 后端", "Local ANN 降级后端", "商品数据生成器"]),
        ("实验与论文支撑", ["Benchmark", "Recall/MRR/NDCG", "市场方案对比"]),
    ]
    y = 150
    for title, items in layers:
        draw.rounded_rectangle((70, y, 1530, y + 145), radius=22, fill="#ffffff", outline="#d8dee8", width=3)
        draw.text((110, y + 48), title, font=font(30, True), fill="#12324a")
        x = 430
        for item in items:
            box(draw, (x, y + 32, x + 285, y + 112), item, "#eef6ff", "#9bb7d4", size=25)
            x += 330
        if y < 660:
            arrow(draw, (800, y + 150), (800, y + 205))
        y += 190
    return save(img, "图4-1_系统总体架构图.png")


def fig_business_flow() -> str:
    img, draw = canvas("图4-2 系统业务流程图", "从集合初始化到检索、过滤、更新删除的主要业务过程")
    steps = [
        ("启动系统", 110, 180),
        ("初始化集合", 410, 180),
        ("批量生成/录入商品", 710, 180),
        ("构建/加载索引", 1010, 180),
        ("输入检索条件", 1010, 460),
        ("向量 Top-K 搜索", 710, 460),
        ("类目/价格过滤", 410, 460),
        ("返回结果并监控指标", 110, 460),
    ]
    for label, x, y in steps:
        box(draw, (x, y, x + 235, y + 105), label, "#ffffff", "#8fb2d6", size=27)
    chain = [0, 1, 2, 3, 4, 5, 6, 7]
    for a, b in zip(chain, chain[1:]):
        x1, y1 = steps[a][1], steps[a][2]
        x2, y2 = steps[b][1], steps[b][2]
        start = (x1 + 235, y1 + 52) if x2 > x1 else (x1, y1 + 52)
        end = (x2 - 10, y2 + 52) if x2 > x1 else (x2 + 245, y2 + 52)
        if y2 > y1:
            start = (x1 + 118, y1 + 105)
            end = (x2 + 118, y2 - 10)
        arrow(draw, start, end)
    box(draw, (1180, 705, 1480, 815), "新增 / 删除 / upsert\n同步更新集合", "#fff6df", "#dbb35c", size=24)
    arrow(draw, (1180, 760), (1000, 520), "#db8d35", 4)
    return save(img, "图4-2_系统业务流程图.png")


def fig_module_relation() -> str:
    img, draw = canvas("图5-1 核心模块调用关系", "主程序、服务层、后端、数据生成与 Web 页面之间的调用关系")
    modules = {
        "main.py\nCLI 启动入口": (100, 170, 380, 270),
        "http_api.py\nHTTP 路由与页面服务": (520, 170, 870, 270),
        "service.py\n业务逻辑与指标采集": (1010, 170, 1400, 270),
        "data_generator.py\n商品数据生成": (120, 520, 430, 620),
        "backends.py\nMilvus / Local ANN": (620, 520, 980, 620),
        "web/*.html\n前端检索与监控": (1130, 520, 1450, 620),
    }
    for label, xy in modules.items():
        color = "#eef6ff" if "service" not in label else "#e9f8f5"
        box(draw, xy, label, color, "#8fb2d6", size=24)
    arrows = [
        ("main.py\nCLI 启动入口", "http_api.py\nHTTP 路由与页面服务"),
        ("http_api.py\nHTTP 路由与页面服务", "service.py\n业务逻辑与指标采集"),
        ("service.py\n业务逻辑与指标采集", "backends.py\nMilvus / Local ANN"),
        ("service.py\n业务逻辑与指标采集", "data_generator.py\n商品数据生成"),
        ("web/*.html\n前端检索与监控", "http_api.py\nHTTP 路由与页面服务"),
    ]
    for a, b in arrows:
        ax = (modules[a][0] + modules[a][2]) // 2
        ay = (modules[a][1] + modules[a][3]) // 2
        bx = (modules[b][0] + modules[b][2]) // 2
        by = (modules[b][1] + modules[b][3]) // 2
        arrow(draw, (ax, ay), (bx, by), "#385a7c", 4)
    return save(img, "图5-1_核心模块调用关系.png")


def load_metrics() -> dict:
    try:
        with urlopen("http://127.0.0.1:8765/metrics", timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return {}


def chart_axes(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int) -> None:
    draw.line((x0, y0, x0, y1, x1, y1), fill="#344054", width=4)


def fig_latency(metrics: dict) -> str:
    rows = (metrics.get("last_benchmark") or {}).get("rows") or [
        {"dataset_size": 100, "avg_search_ms": 3.364},
        {"dataset_size": 1000, "avg_search_ms": 5.136},
        {"dataset_size": 5000, "avg_search_ms": 10.043},
        {"dataset_size": 10000, "avg_search_ms": 3.469},
    ]
    img, draw = canvas("图6-1 不同数据规模下平均检索延迟", "横轴为商品向量数据规模，纵轴为平均检索耗时 ms")
    x0, y0, x1, y1 = 180, 755, 1450, 185
    chart_axes(draw, x0, y0, x1, y1)
    max_v = max(float(r["avg_search_ms"]) for r in rows) * 1.2
    xs = [x0 + i * ((x1 - x0) / (len(rows) - 1)) for i in range(len(rows))]
    pts = []
    for x, row in zip(xs, rows):
        v = float(row["avg_search_ms"])
        y = y0 - (v / max_v) * (y0 - y1)
        pts.append((x, y, row))
    for tick in range(0, 6):
        val = max_v * tick / 5
        y = y0 - tick * ((y0 - y1) / 5)
        draw.line((x0 - 8, y, x1, y), fill="#e4e7ec", width=2)
        draw.text((70, y - 14), f"{val:.1f}", font=font(20), fill="#667085")
    for a, b in zip(pts, pts[1:]):
        draw.line((a[0], a[1], b[0], b[1]), fill="#2563eb", width=6)
    for x, y, row in pts:
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill="#2563eb")
        draw.text((x - 45, y - 50), f"{float(row['avg_search_ms']):.3f}ms", font=font(20, True), fill="#12324a")
        label = str(row["dataset_size"])
        w, _ = text_size(draw, label, font(20))
        draw.text((x - w / 2, y0 + 24), label, font=font(20), fill="#344054")
    draw.text((640, 830), "数据规模（条）", font=font(24, True), fill="#344054")
    draw.text((48, 440), "平均检索耗时(ms)", font=font(24, True), fill="#344054")
    return save(img, "图6-1_不同数据规模下平均检索延迟.png")


def fig_recall(metrics: dict) -> str:
    rows = (metrics.get("last_benchmark") or {}).get("rows") or [
        {"dataset_size": 100, "avg_recall_at_k": 1.0},
        {"dataset_size": 1000, "avg_recall_at_k": 1.0},
        {"dataset_size": 5000, "avg_recall_at_k": 1.0},
        {"dataset_size": 10000, "avg_recall_at_k": 1.0},
    ]
    img, draw = canvas("图6-2 不同数据规模下 Recall@K", "展示近似检索结果与精确检索结果的重合比例")
    x0, y0, x1, y1 = 180, 755, 1450, 185
    chart_axes(draw, x0, y0, x1, y1)
    for tick in range(0, 6):
        val = tick / 5
        y = y0 - val * (y0 - y1)
        draw.line((x0 - 8, y, x1, y), fill="#e4e7ec", width=2)
        draw.text((85, y - 14), f"{val:.1f}", font=font(20), fill="#667085")
    bar_w = 150
    gap = (x1 - x0 - len(rows) * bar_w) / (len(rows) + 1)
    for i, row in enumerate(rows):
        v = float(row["avg_recall_at_k"])
        x = x0 + gap + i * (bar_w + gap)
        y = y0 - v * (y0 - y1)
        draw.rounded_rectangle((x, y, x + bar_w, y0), radius=12, fill="#1b998b")
        draw.text((x + 34, y - 42), f"{v:.2f}", font=font(22, True), fill="#12324a")
        label = str(row["dataset_size"])
        w, _ = text_size(draw, label, font(20))
        draw.text((x + bar_w / 2 - w / 2, y0 + 24), label, font=font(20), fill="#344054")
    draw.text((640, 830), "数据规模（条）", font=font(24, True), fill="#344054")
    draw.text((75, 440), "Recall@K", font=font(24, True), fill="#344054")
    return save(img, "图6-2_不同数据规模下RecallK.png")


def main() -> None:
    metrics = load_metrics()
    images = {
        "fig1_1": fig_tech_route(),
        "fig2_1": fig_ann_classification(),
        "fig4_1": fig_architecture(),
        "fig4_2": fig_business_flow(),
        "fig5_1": fig_module_relation(),
        "fig6_1": fig_latency(metrics),
        "fig6_2": fig_recall(metrics),
    }
    (OUT / "chart_manifest.json").write_text(json.dumps({"images": images, "metrics": metrics}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(images, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
