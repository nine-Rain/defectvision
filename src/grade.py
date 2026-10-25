"""缺陷等级判定层：按开题报告定义的四级规则给缺陷定级。

规则来源是开题报告（2026-09-23）"缺陷等级判定方法研究"一节，不是拍脑袋定的：

    I 级（轻微）：面积 ≤0.5%、数量 ≤2 且不含裂纹 —— 合格放行
    II 级（中度）：面积 0.5%~2%，或数量 3~5，或非贯穿性短裂纹 —— 让步接收
    III 级（严重）：面积 2%~5%，或数量 6~10，或裂纹超长 —— 返工重检
    IV 级（致命）：面积 >5%，或连片密集，或贯穿性裂纹 —— 判废追溯

判定原则是"就高不就低"：四个判据各自给级，取最高者。任何单一指标恶化都不会
被其他指标平均掉，也便于向工艺人员逐条解释——这是纯评分模型做不到的。

    from src.grade import grade
    result = grade(detections, image_size)
    print(result.level_name, result.action)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from src.config import repo_path
from src.detectors import Detection


@dataclass
class GradeResult:
    level: int
    level_name: str
    action: str
    score: float
    features: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    defects: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "level_name": self.level_name,
            "action": self.action,
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


def _union_area_ratio(boxes: list[tuple[float, float, float, float]], width: int, height: int) -> float:
    """把框画到布尔掩码上求并集，避免重叠框把面积重复计入。"""
    mask = np.zeros((height, width), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        left = int(_clamp(x0, 0, width - 1))
        top = int(_clamp(y0, 0, height - 1))
        right = int(_clamp(x1, left + 1, width))
        bottom = int(_clamp(y1, top + 1, height))
        mask[top:bottom, left:right] = True
    return float(mask.mean())


def _box_gap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


def _level_by_area(ratio: float, limits: dict) -> int:
    if ratio > limits["level4"]:
        return 4
    if ratio > limits["level3"]:
        return 3
    if ratio > limits["level2"]:
        return 2
    return 1


def _level_by_count(count: int, limits: dict) -> int:
    if count >= limits["level4"]:
        return 4
    if count >= limits["level3"]:
        return 3
    if count >= limits["level2"]:
        return 2
    return 1


def grade(
    detections: list[Detection],
    image_size: tuple[int, int],
    config: dict | None = None,
    conf_floor: float = 0.25,
) -> GradeResult:
    """image_size = (width, height)。置信度低于 conf_floor 的检测不参与判级。"""
    cfg = config or load_grading_config()
    levels: dict[int, dict] = {int(k): v for k, v in cfg["levels"].items()}
    criteria = cfg["criteria"]
    severity_map: dict[str, int] = cfg.get("class_severity", {})

    width, height = image_size
    # 裂纹长度按"图像长边"归一化。用对角线会有个隐藏问题：
    # 轴对齐框的最大边长最多只到 0.707 倍对角线，阈值 0.8 永远触发不了。
    max_dim = float(max(width, height))
    diagonal = math.hypot(width, height)
    kept = [d for d in detections if d.confidence >= conf_floor]

    if not kept:
        return GradeResult(
            level=0,
            level_name="合格（未检出缺陷）",
            action="直接放行",
            score=0.0,
            features={"area_ratio": 0.0, "count": 0.0, "severity": 0.0, "spread": 0.0},
            reasons=["未检出缺陷，判定合格"],
            defects=[],
        )

    # ---------- 特征量化 ----------
    boxes = [d.box for d in kept]
    area_ratio = _union_area_ratio(boxes, width, height)
    count = len(kept)
    severity_max = max(severity_map.get(d.cls_name, 2) for d in kept)

    union = [min(b[0] for b in boxes), min(b[1] for b in boxes),
             max(b[2] for b in boxes), max(b[3] for b in boxes)]
    spread = _clamp(((union[2] - union[0]) * (union[3] - union[1])) / (width * height))

    crack_cfg = criteria["crack"]
    crack_names = {str(name).lower() for name in crack_cfg.get("classes", [])}
    crack_lengths = []
    for detection in kept:
        if detection.cls_name.lower() not in crack_names:
            continue
        box_width = detection.box[2] - detection.box[0]
        box_height = detection.box[3] - detection.box[1]
        crack_lengths.append(max(box_width, box_height) / max_dim)

    clustered = False
    cluster_cfg = criteria["clustered"]
    if count >= cluster_cfg["min_count"]:
        gaps = [
            _box_gap(boxes[i], boxes[j])
            for i in range(len(boxes))
            for j in range(i + 1, len(boxes))
        ]
        clustered = bool(gaps) and min(gaps) <= cluster_cfg["max_gap_ratio"] * diagonal

    # ---------- 四个判据各自给级（就高不就低） ----------
    candidates: list[tuple[str, int]] = []
    candidates.append(("面积占比", _level_by_area(area_ratio, criteria["area_ratio"])))
    candidates.append(("缺陷数量", _level_by_count(count, criteria["count"])))

    if crack_lengths:
        longest = max(crack_lengths)
        if longest >= crack_cfg["length_level4_ratio"]:
            candidates.append(("贯穿性裂纹", 4))
        elif longest >= crack_cfg["length_level3_ratio"]:
            candidates.append(("裂纹长度超限", 3))
        elif crack_cfg.get("short_level2", True):
            candidates.append(("存在非贯穿性短裂纹", 2))

    if clustered:
        candidates.append(("缺陷连片密集", 4))

    final_level = max(level for _, level in candidates)
    final = levels[final_level]

    # ---------- 只保留真正决定等级的那几条理由 ----------
    reasons: list[str] = [
        f"检出 {count} 处缺陷，覆盖面积 {area_ratio * 100:.2f}%"
    ]
    for name, level in candidates:
        if level >= final_level:
            reasons.append(f"{name}判据达到 {levels[level]['name']}")
    reasons.append(f"按“就高不就低”原则，最终定为 {final['name']}")

    strongest = max(kept, key=lambda d: severity_map.get(d.cls_name, 2))
    reasons.append(f"最严重缺陷类型为 {strongest.cls_name}")

    # ---------- 综合得分（仅展示，不参与定级） ----------
    score_cfg = cfg.get("score", {})
    weights = score_cfg.get("weights", {})
    normalizers = score_cfg.get("normalizers", {})
    normalized = {
        "area": _clamp(area_ratio / normalizers.get("area_ratio", 0.05)),
        "count": _clamp(count / normalizers.get("count", 10)),
        "severity": _clamp(severity_max / 4.0),
        "spread": _clamp(spread / normalizers.get("spread", 0.5)),
    }
    score = 100.0 * sum(weights.get(key, 0.0) * normalized[key] for key in normalized)

    defects = [
        {
            "class": d.cls_name,
            "confidence": round(d.confidence, 3),
            "area_ratio": round(d.area / (width * height), 4),
            "severity": severity_map.get(d.cls_name, 2),
        }
        for d in kept
    ]

    features = {
        "area_ratio": area_ratio,
        "count": float(count),
        "severity": float(severity_max),
        "spread": spread,
        "crack_max_length": max(crack_lengths) if crack_lengths else 0.0,
        "clustered": 1.0 if clustered else 0.0,
    }
    return GradeResult(
        level=final_level,
        level_name=final["name"],
        action=final["action"],
        score=score,
        features=features,
        reasons=reasons,
        defects=defects,
    )


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
        f"处置意见: {result.action}",
        f"综合得分: {result.score:.1f} / 100（仅供参考）",
        "-" * 46,
        "量化特征:",
        f"  缺陷数量      {int(result.features.get('count', 0))}",
        f"  覆盖面积占比  {result.features.get('area_ratio', 0) * 100:.2f}%",
        f"  最高严重度    {int(result.features.get('severity', 0))} / 4",
        f"  分布覆盖      {result.features.get('spread', 0) * 100:.1f}%",
    ]
    if result.features.get("crack_max_length"):
        lines.append(f"  最长裂纹      {result.features['crack_max_length'] * 100:.1f}% 图像长边")
    if result.features.get("clustered"):
        lines.append("  连片密集      是")
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
