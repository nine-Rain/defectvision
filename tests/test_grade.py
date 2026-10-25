"""缺陷判级规则的测试。规则来自开题报告，纯逻辑，不需要 torch。

四个判据（面积 / 数量 / 裂纹 / 连片）各自给级，取最高——"就高不就低"。
"""

from __future__ import annotations

import pytest

from src.detectors import Detection
from src.grade import format_report, grade, load_grading_config

IMAGE_SIZE = (512, 512)  # 262,144 像素


def box(x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
    return (x0, y0, x1, y1)


def detection(
    cls_name: str,
    bbox: tuple[float, float, float, float],
    cls_id: int = 0,
    conf: float = 0.9,
) -> Detection:
    return Detection(cls_id=cls_id, cls_name=cls_name, confidence=conf, box=bbox)


def square(size: int, left: int = 0, top: int = 0) -> tuple[float, float, float, float]:
    return box(float(left), float(top), float(left + size), float(top + size))


def test_config_matches_proposal_levels():
    cfg = load_grading_config()
    assert sorted(int(k) for k in cfg["levels"]) == [1, 2, 3, 4]
    assert cfg["criteria"]["area_ratio"] == {"level2": 0.005, "level3": 0.020, "level4": 0.050}
    assert cfg["criteria"]["count"] == {"level2": 3, "level3": 6, "level4": 11}


def test_no_detection_is_ok():
    result = grade([], IMAGE_SIZE)
    assert result.level == 0
    assert result.level_name.startswith("合格")
    assert result.action == "直接放行"


def test_tiny_non_crack_defect_is_level1():
    """面积 0.15%、数量 1、非裂纹类 -> I 级。"""
    result = grade([detection("stain", square(20))], IMAGE_SIZE)
    assert result.level == 1
    assert result.action.startswith("合格")


def test_area_criterion_level2():
    """50x50 = 0.95% -> 面积判据 II 级。用非裂纹类，避免裂纹判据干扰。"""
    result = grade([detection("stain", square(50))], IMAGE_SIZE)
    assert result.level == 2


def test_area_criterion_level3():
    """100x100 = 3.8% -> III 级。"""
    result = grade([detection("stain", square(100))], IMAGE_SIZE)
    assert result.level == 3


def test_area_criterion_level4():
    """150x150 = 8.6% -> IV 级。"""
    result = grade([detection("stain", square(150))], IMAGE_SIZE)
    assert result.level == 4


def test_count_criterion_level3():
    """6 处小缺陷：面积很小，但数量到了 6 -> III 级。"""
    spots = [detection("stain", square(20, left=index * 70)) for index in range(6)]
    result = grade(spots, IMAGE_SIZE)
    assert result.level == 3
    assert any("缺陷数量" in reason for reason in result.reasons)


def test_crack_short_triggers_level2():
    """存在裂纹类缺陷（哪怕很短）-> 至少 II 级。"""
    result = grade([detection("scratch", square(20))], IMAGE_SIZE)
    assert result.level == 2
    assert any("短裂纹" in reason for reason in result.reasons)


def test_crack_long_triggers_level3():
    """裂纹长度 400/512 = 78% -> 超限，III 级。"""
    result = grade([detection("scratch", box(0, 250, 400, 260))], IMAGE_SIZE)
    assert result.level == 3
    assert any("裂纹长度超限" in reason for reason in result.reasons)


def test_crack_through_triggers_level4():
    """裂纹长度 420/512 = 82% -> 视为贯穿性，IV 级。"""
    result = grade([detection("scratch", box(0, 250, 420, 258))], IMAGE_SIZE)
    assert result.level == 4
    assert any("贯穿性" in reason for reason in result.reasons)


def test_clustered_defects_trigger_level4():
    """4 个缺陷彼此相邻（间距 5px < 2% 对角线）-> 连片密集，IV 级。"""
    spots = [detection("stain", square(20, left=index * 25)) for index in range(4)]
    result = grade(spots, IMAGE_SIZE)
    assert result.level == 4
    assert any("连片" in reason for reason in result.reasons)


def test_take_the_highest_criterion():
    """数量只到 II，但面积到 III -> 取 III。"""
    spots = [detection("stain", square(100, left=10)), detection("stain", square(20, left=400))]
    result = grade(spots, IMAGE_SIZE)
    assert result.level == 3
    assert any("就高不就低" in reason for reason in result.reasons)


def test_low_confidence_is_ignored():
    result = grade([detection("stain", square(150), conf=0.1)], IMAGE_SIZE, conf_floor=0.25)
    assert result.level == 0


def test_overlapping_boxes_are_not_double_counted():
    """两个几乎重合的大框，面积占比不应翻倍到 IV 级。"""
    first = detection("stain", square(100))
    second = detection("stain", square(100, left=2, top=2))
    result = grade([first, second], IMAGE_SIZE)
    # 并集约 102x102 = 3.97%，落在 III；如果简单相加会到 7.7% 变成 IV
    assert result.level == 3


def test_criteria_are_reported_per_rule():
    """界面要能逐条解释"为什么判 IV 级"，所以每个判据的等级都要带出来。"""
    result = grade([detection("scratch", box(0, 250, 420, 258))], IMAGE_SIZE)
    assert result.level == 4
    assert result.criteria["面积占比"] == 2      # 只有 1.3%，面积判据只到 II
    assert result.criteria["缺陷数量"] == 1      # 只有 1 处
    assert result.criteria["贯穿性裂纹"] == 4    # 长度 82% >= 80%，这条把它拉到 IV
    assert max(result.criteria.values()) == result.level


def test_defect_table_marks_type_baseline_not_final_level():
    """单个缺陷的 severity 只是类型基准，不能等于最终等级。"""
    result = grade([detection("scratch", box(0, 250, 420, 258))], IMAGE_SIZE)
    assert result.defects[0]["severity"] == 2   # scratch 的类型基准
    assert result.level == 4                    # 但最终等级由判据决定


def test_report_contains_level_action_and_reasons():
    result = grade([detection("stain", square(100))], IMAGE_SIZE)
    report = format_report(result, image_name="a.jpg", model_name="best.pt", elapsed_ms=12.3)
    assert "表面缺陷检测报告" in report
    assert result.level_name in report
    assert result.action in report
    assert "判定依据" in report
    assert "12.3 ms" in report


@pytest.mark.parametrize("size", [40, 80, 120, 200])
def test_score_grows_with_area(size: int):
    small = grade([detection("stain", square(max(10, size // 2)))], IMAGE_SIZE).score
    big = grade([detection("stain", square(size))], IMAGE_SIZE).score
    assert big >= small
