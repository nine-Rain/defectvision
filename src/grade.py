"""缺陷等级判定层：把检测结果映射成 I~IV 级，并生成可解释的报告。

设计原则：
1. 规则可配置（configs/grading.yaml），工艺人员改阈值不用重训模型；
2. 每个等级都要给出理由，而不是只丢一个数字——产线上没人信黑盒判级；
3. 检测置信度低的结果要降权，避免一个误检把整批判成报废。

    from src.grade import grade
    result = grade(detections, image_size)
    print(result.level_name, result.score)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.config import repo_path
from src.detectors import Detection


@dataclass
class GradeResult:
    level: int
    level_name: str
    score: float
    features: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    defects: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "level_name": self.level_name,
            "score": round(self.score, 2),
            "features": {k: round(v, 4) for k, v in self.features.items()},
            "reasons": self.reasons,
            "defects": self.defects,
        }


def load_grading_config(path: str | Path = "configs/grading.yaml") -> dict:
    resolved = repo_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"找不到判级配置: {resolved}")
    return yaml.safe_load(resolved.read_text(encoding="utf-8"))


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def grade(
    detections: list[Detection],
    image_size: tuple[int, int],
    config: dict | None = None,
    conf_floor: float = 0.25,
) -> GradeResult:
    """image_size = (width, height)。置信度低于 conf_floor 的检测不参与判级。"""
    cfg = config or load_grading_config()
    severity_map: dict[str, int] = cfg.get("class_severity", {})
    weights: dict[str, float] = cfg["weights"]
    normalizers: dict[str, float] = cfg["normalizers"]
    levels: list[dict] = cfg["levels"]

    width, height = image_size
    image_area = max(1.0, float(width * height))

    kept = [d for d in detections if d.confidence >= conf_floor]
    defects: list[dict] = []
    area_sum = 0.0
    severity_max = 0
    union = [float("inf"), float("inf"), 0.0, 0.0]

    for detection in kept:
        severity = severity_map.get(detection.cls_name, 2)
        severity_max = max(severity_max, severity)
        area_sum += detection.area
        x0, y0, x1, y1 = detection.box
        union[0] = min(union[0], x0)
        union[1] = min(union[1], y0)
        union[2] = max(union[2], x1)
        union[3] = max(union[3], y1)
        defects.append(
            {
                "class": detection.cls_name,
                "confidence": round(detection.confidence, 3),
                "area_ratio": round(detection.area / image_area, 4),
                "severity": severity,
            }
        )

    area_ratio = area_sum / image_area
    count = len(kept)
    spread = 0.0
    if count:
        spread = max(0.0, (union[2] - union[0]) * (union[3] - union[1])) / image_area

    features = {
        "area_ratio": area_ratio,
        "count": float(count),
        "severity": float(severity_max),
        "spread": spread,
    }
    normalized = {
        "area": _clamp(area_ratio / normalizers["area_ratio"]),
        "count": _clamp(count / normalizers["count"]),
        "severity": _clamp(severity_max / 4.0),
        "spread": _clamp(spread / normalizers["spread"]),
    }
    score = 100.0 * sum(weights[key] * normalized[key] for key in weights)

    level_index = len(levels) - 1
    for index, entry in enumerate(levels):
        if score < entry["upper"]:
            level_index = index
            break
    level = levels[level_index]["level"]
    level_name = levels[level_index]["name"]

    reasons: list[str] = []
    if count == 0:
        reasons.append("未检出缺陷，判为合格")
        return GradeResult(0, "合格", 0.0, features, reasons, defects)

    reasons.append(f"检出 {count} 处缺陷，总面积占比 {area_ratio * 100:.2f}%")
    strongest = max(defects, key=lambda d: d["severity"])
    reasons.append(f"最严重缺陷类型为 {strongest['class']}（严重度 {strongest['severity']}）")
    if spread > 0.5:
        reasons.append("缺陷分布分散，疑似多工序引入")

    # 一票否决：关键类型 + 面积不小，直接提到指定等级
    veto = cfg.get("veto") or {}
    if veto and count:
        big_severe = [
            d for d in defects
            if d["severity"] >= veto.get("severity_gte", 5)
            and d["area_ratio"] >= veto.get("min_area_ratio", 1.0)
        ]
        if big_severe and level < veto.get("forced_level", 4):
            level = veto["forced_level"]
            level_name = levels[min(level - 1, len(levels) - 1)]["name"]
            reasons.append(
                f"触发一票否决：{big_severe[0]['class']} 面积 {big_severe[0]['area_ratio'] * 100:.2f}%，"
                f"直接判 {level_name}"
            )

    return GradeResult(level, level_name, score, features, reasons, defects)


def format_report(
    result: GradeResult,
    image_name: str = "",
    model_name: str = "",
    elapsed_ms: float | None = None,
) -> str:
    """生成纯文本报告，桌面端可以直接存成 .txt。"""
    lines = [
        "=" * 46,
        "        表面缺陷检测报告",
        "=" * 46,
        f"图像    : {image_name or '(未命名)'}",
        f"模型    : {model_name or '(未知)'}",
    ]
    if elapsed_ms is not None:
        lines.append(f"耗时    : {elapsed_ms:.1f} ms")
    lines += [
        "-" * 46,
        f"判定    : {result.level_name}",
        f"综合得分: {result.score:.1f} / 100",
        "-" * 46,
        "量化特征:",
        f"  缺陷数量      {int(result.features.get('count', 0))}",
        f"  面积占比      {result.features.get('area_ratio', 0) * 100:.2f}%",
        f"  最高严重度    {int(result.features.get('severity', 0))} / 4",
        f"  分布覆盖      {result.features.get('spread', 0) * 100:.1f}%",
    ]
    if result.defects:
        lines.append("-" * 46)
        lines.append("缺陷清单:")
        for index, defect in enumerate(result.defects, start=1):
            lines.append(
                f"  {index}. {defect['class']:<12} 置信度 {defect['confidence']:.3f}"
                f"  面积 {defect['area_ratio'] * 100:.2f}%  严重度 {defect['severity']}"
            )
    if result.reasons:
        lines.append("-" * 46)
        lines.append("判定依据:")
        for reason in result.reasons:
            lines.append(f"  · {reason}")
    lines.append("=" * 46)
    return "\n".join(lines)
