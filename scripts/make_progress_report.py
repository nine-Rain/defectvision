"""生成《阶段性进展汇报》DOCX。

    python scripts/make_progress_report.py

数据来源：results/ 下的实验记录 + docs/review/index.csv 的人工复核结果。
改数就改结论表，重新跑一遍即可，避免文档与实验脱节。
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

OUT = Path(__file__).resolve().parent.parent / "docs" / "阶段性进展汇报.docx"


def set_font(run, name="宋体", size=10.5, bold=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)


def para(doc, text, size=10.5, bold=False, align=None, space_after=4):
    p = doc.add_paragraph()
    run = p.add_run(text)
    set_font(run, size=size, bold=bold)
    if align:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    return p


def heading(doc, text, level=1):
    p = doc.add_paragraph()
    run = p.add_run(text)
    set_font(run, name="黑体", size=14 if level == 1 else 12, bold=True)
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(6)
    return p


def bullet(doc, text, size=10.5):
    p = doc.add_paragraph(style="List Bullet")
    run = p.add_run(text)
    set_font(run, size=size)
    p.paragraph_format.space_after = Pt(2)
    return p


def table(doc, headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        set_font(run, size=9, bold=True)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(value))
            set_font(run, size=9)
            cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    return t


def main() -> None:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "宋体"
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    title = doc.add_paragraph()
    run = title.add_run("本科毕业论文阶段性进展汇报")
    set_font(run, name="黑体", size=18, bold=True)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph()
    run = sub.add_run("论文题目：基于改进 CNN 的工业零件表面缺陷检测与等级判定系统设计与实现")
    set_font(run, size=10.5)
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

    info = doc.add_paragraph()
    run = info.add_run("学生：刘权威    学号：202323140622    学院：人工智能学院    指导教师：刘锡铃")
    set_font(run, size=9, color=(90, 90, 90))
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ---------------- 一 ----------------
    heading(doc, "一、总体进展")
    para(
        doc,
        "开题时计划的技术路线（改进 CNN 检测 + 四级等级判定 + PyQt5 系统）已全部实现并跑通，"
        "并完成了开题答辩中老师提出的两项补充：与成熟的现有检测方法做对比、补充量化分级实验。"
        "近期又额外做了一轮人工复核，用来确认分级指标的真实含义。",
    )
    table(
        doc,
        ["模块", "状态", "说明"],
        [
            ["数据集", "已完成", "MSDD（1195/150/150）+ NEU-DET（1458/162/180），已校验无缺失标注"],
            ["改进 CNN 模型", "已完成", "ResNet50 + FPN + CBAM，含无 CBAM 消融"],
            ["对比实验", "已完成", "YOLO11n / YOLOv8s / YOLO11n-seg / RT-DETR-l / 自研 CNN，含精度、体积、延迟"],
            ["四级等级判定", "已完成", "严格按开题定义实现（就高不就低），含 29 个单元测试"],
            ["量化分级实验", "已完成", "一致率、±1 一致率、IV 级召回、误杀率、混淆矩阵"],
            ["人工复核", "已完成", "42 张争议样本逐张判读，修正了对误杀原因的判断"],
            ["PyQt5 系统", "已完成", "单图检测/批量检测/双模型对比/报告导出"],
            ["自建数据集", "进行中", "采集流水线已就绪，已收集 11 张种子图，待补拍 60+"],
        ],
    )

    # ---------------- 二 ----------------
    heading(doc, "二、检测实验结果（MSDD，test 集）")
    table(
        doc,
        ["模型", "mAP50", "mAP50-95", "参数量", "体积", "p50 延迟"],
        [
            ["RT-DETR-l @512", "0.641", "0.427", "32.8 M", "63.1 MB", "22.5 ms"],
            ["YOLOv8s @512", "0.588", "0.376", "11.1 M", "21.5 MB", "12.0 ms"],
            ["YOLO11n @512", "0.559", "0.384", "2.6 M", "5.2 MB", "12.3 ms"],
            ["YOLO11n @640", "0.590", "0.387", "2.6 M", "5.3 MB", "14.5 ms"],
            ["YOLO11n-seg（掩码）", "0.580", "0.343", "2.8 M", "5.8 MB", "—"],
            ["自研 ResNet50+FPN+CBAM", "0.574", "0.298", "41.4 M", "158.3 MB", "34.3 ms"],
        ],
    )
    para(
        doc,
        "结论：自研模型的 mAP50 与 YOLO 系列接近（0.574 vs 0.559），但 mAP50-95 明显偏低（0.298 vs 0.384），"
        "即“找得到缺陷、贴不紧边界”；同时参数量与体积大一个数量级、推理慢约 3 倍。"
        "检测精度最高的是 RT-DETR（0.641 / 0.427），代价是 63 MB 与 22.5 ms。",
        size=9,
    )

    # ---------------- 三 ----------------
    heading(doc, "三、消融实验（CBAM 是否有效）")
    table(
        doc,
        ["配置", "轮数/增强", "val mAP50", "test mAP50", "test mAP50-95"],
        [
            ["ResNet50+FPN", "20 轮 / 无", "0.453", "0.494", "0.262"],
            ["+ CBAM", "20 轮 / 无", "0.435", "0.476", "0.248"],
            ["ResNet50+FPN", "50 轮 / 有", "0.517", "0.516", "0.275"],
            ["+ CBAM", "50 轮 / 有", "0.507", "0.574", "0.298"],
        ],
    )
    para(
        doc,
        "结论：CBAM 的效应不稳定——20 轮时无 CBAM 更好，50 轮带增强时 val 与 test 结论相反。"
        "差异落在 150 张验证集的抽样波动范围内，不足以支撑“CBAM 有效”的结论。"
        "需要与老师讨论如何调整创新点的表述。",
        size=9,
    )

    # ---------------- 四 ----------------
    heading(doc, "四、量化分级实验（开题答辩建议补充的部分）")
    para(
        doc,
        "方法：用标注按同一套规则算出“参考等级”（固定不变），与模型判出的“系统等级”比较。"
        "MSDD test，150 张，置信度阈值 0.25。",
        size=9,
    )
    table(
        doc,
        ["模型", "面积口径", "完全一致率", "±1 级内", "IV 级召回", "低等级误杀率"],
        [
            ["RT-DETR-l", "检测框", "81.3%", "83.3%", "99.0%", "28.1%"],
            ["YOLOv8s", "检测框", "83.3%", "86.7%", "93.9%", "6.2%"],
            ["YOLO11n", "检测框", "82.7%", "84.7%", "92.9%", "9.4%"],
            ["YOLO11n-seg", "掩码像素", "79.3%", "87.3%", "93.2%", "5.3%"],
            ["自研 CNN（50轮+增强）", "检测框", "73.3%", "75.3%", "94.9%", "59.4%"],
            ["自研 CNN（20轮）", "检测框", "65.3%", "66.7%", "98.0%", "93.8%"],
        ],
    )
    para(
        doc,
        "关键发现：检测精度最高的 RT-DETR 并不是判级最好的（81.3%，低于 YOLOv8s 的 83.3%）；"
        "自研 CNN 的 mAP50 高于 YOLO11n，判级一致率却低 9 个百分点、误杀率是它的 6 倍。"
        "说明检测指标不能代表系统可用性，分级实验有必要单独做。",
        size=9,
    )

    # ---------------- 五 ----------------
    heading(doc, "五、优化实验：五种尝试，两种有效")
    table(
        doc,
        ["手段", "对检测的影响", "对判级的影响", "结论"],
        [
            ["判级阈值校准（216 组网格搜索）", "—", "±0.7 个百分点", "无效"],
            ["判级前 NMS 后处理", "—", "误杀 28.1%→28.1%", "无效（不是重复框）"],
            ["面积按置信度加权", "—", "一致率 −3.3 点", "无效，反而更差"],
            ["过滤超大面积整体框", "—", "误杀 ↓ 但 IV 召回 99%→64%", "不可用"],
            ["面积口径改为掩码像素", "—", "误杀 9.4%→5.3%", "有效，已设为默认"],
            ["置信度工作点标定（0.25→0.55）", "—", "RT-DETR 误杀 28.1%→3.1%", "有效"],
            ["输入分辨率 512→640", "mAP50 +3.1 点", "一致率 −8 点", "得不偿失"],
        ],
    )
    para(
        doc,
        "这轮实验的价值在于“排除”：判级公式层能改的地方基本试完了，"
        "剩下的问题不在规则，而在检测输出的框本身。",
        size=9,
    )

    # ---------------- 六 ----------------
    heading(doc, "六、人工复核：42 张争议样本")
    para(
        doc,
        "把“判级比标注高 2 级及以上”的样本全部挑出来，拼成复核图，由学生与外部协助者分别独立判读，"
        "结果合并如下：",
        size=9,
    )
    table(
        doc,
        ["判定", "数量", "占比", "含义"],
        [
            ["模型误检", "39", "93%", "该处确实没有缺陷，模型多检了"],
            ["标注漏标", "3", "7%", "图上有缺陷但标注没标，模型其实是对的"],
            ["存疑", "0", "0%", "—"],
        ],
    )
    para(
        doc,
        "两个人都观察到的失败模式：81%（34/42）的样本里，模型输出框的总面积超过图像 40%，"
        "几乎盖住整个零件；而且这些图大多确实存在真实缺陷——模型只是在此之外多画了一层大框，"
        "把覆盖面积撑到 100% 以上，等级因此被顶到 IV。",
        size=9,
    )
    para(
        doc,
        "结论：误杀是真实存在的，不能归因于标注噪声；但数据集中确实存在约 7% 的漏标，"
        "因此论文中相关的表述应写作“在此标注下的一致率”，而不是“判级准确率”。",
        size=9,
        bold=True,
    )

    # ---------------- 七 ----------------
    heading(doc, "七、系统实现（PyQt5）")
    bullet(doc, "单图检测：导入图片后自动检测、判级，显示等级、处置意见、四个判据的分别定级、缺陷清单。")
    bullet(doc, "批量检测：整个文件夹一次处理，输出每张图的报告与一份 CSV 汇总（按等级可筛）。")
    bullet(doc, "双模型对比：同一张图左右并排跑两个模型，直接显示等级差异。")
    bullet(doc, "报告导出：文本报告含量化特征、判据明细、判定依据，并标注面积口径（掩码/检测框）。")

    # ---------------- 八 ----------------
    heading(doc, "八、目前存在的不足")
    items = [
        ("自建数据集仍在补充中", "采集流水线已就绪（视频抽帧/图片收集/来源记录），目前 11 张种子图，待补拍至 60 张以上。"),
        ("CBAM 未带来稳定提升", "消融实验结论摇摆，创新点“有效提升精度”的表述缺乏支撑，需要调整。"),
        ("自研模型定位精度偏低", "mAP50-95 仅 0.298，框贴不紧边界；其判级误杀率高达 59%，尚不具备产线可用性。"),
        ("判级公式层已无优化空间", "阈值校准、NMS、置信度加权、整体框过滤均无效，说明瓶颈在检测输出而非规则。"),
        ("标注存在约 7% 漏标", "人工复核 42 张发现 3 张标注漏标；这会同时抬高参考等级与模型等级，使两者都失真。"),
        ("分级缺少人工真值", "参考等级仍由规则从标注推导，尚未经检验人员独立判定等级。"),
        ("统计功效有限", "判级实验仅 150 张、单次运行，未做多种子重复与置信区间。"),
        ("对比模型仍需扩展", "已补 RT-DETR-l；Transformer 系目前仅此一个模型。"),
    ]
    for name, detail in items:
        p = doc.add_paragraph(style="List Number")
        run = p.add_run(f"{name}：")
        set_font(run, size=10.5, bold=True)
        run2 = p.add_run(detail)
        set_font(run2, size=10.5)
        p.paragraph_format.space_after = Pt(3)

    # ---------------- 九 ----------------
    heading(doc, "九、请老师指示的问题")
    questions = [
        "创新点如何调整？CBAM 未带来稳定提升，是否可将创新点改为“检测+分级一体化框架与消融分析”，而非“提升精度”？",
        "分级参考等级是否需要人工标注？现有证据表明规则推导的标准与模型共享“框面积”语义，容易自证；建议标 50~100 张。",
        "自建数据集的规模要求？建议拍摄 60~100 张、2~3 类缺陷，是否满足毕业论文的泛化性验证要求？",
        "是否需要针对“整体框”误检做一次难负样本重训？预计 1 小时，但收益不确定。",
        "论文重点应放在检测模型改进，还是分级方法与系统实现？目前数据表明后两者更有差异化价值。",
    ]
    for index, question in enumerate(questions, start=1):
        p = doc.add_paragraph()
        run = p.add_run(f"{index}. {question}")
        set_font(run, size=10.5)
        p.paragraph_format.space_after = Pt(4)

    # ---------------- 附 ----------------
    heading(doc, "附：代码与实验记录")
    para(doc, "代码仓库：https://github.com/nine-Rain/defectvision", size=9)
    para(doc, "所有指标由脚本自动生成并归档于 results/；人工复核记录见 docs/review/。", size=9)
    para(doc, "包含 47 个单元测试与持续集成检查；实验环境 Python 3.12 + PyTorch 2.14 + CUDA 12.6，RTX 4060 Laptop 8G。", size=9)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT))
    print(f"已生成: {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
