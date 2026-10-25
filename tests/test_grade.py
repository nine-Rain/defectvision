"""缺陷判级规则的测试。纯规则，不需要 torch / 显卡。"""

from __future__ import annotations

import pytest

from src.detectors import Detection
from src.grade import format_report, grade, load_grading_config

IMAGE_SIZE = (512, 512)


def make_box(size: int, left: int = 0, top: int = 0) -> tuple[float, float, float, float]:
    return (float(left), float(top), float(left + size), float(top + size))


def detection(
    cls_name: str, cls_id: int = 0, size: int = 50, conf: float = 0.9, left: int = 0, top: int = 0
) -> Detection:
    return Detection(cls_id=cls_id, cls_name=cls_name, confidence=conf, box=make_box(size, left, top))


def test_config_loads():
    cfg = load_grading_config()
    assert "weights" in cfg and "levels" in cfg
    assert abs(sum(cfg["weights"].values()) - 1.0) < 1e-6


def test_no_detection_is_ok():
    result = grade([], IMAGE_SIZE)
    assert result.level == 0
    assert result.level_name == "合格"
    assert result.score == 0


def test_single_small_scratch_is_minor():
    result = grade([detection("scratch")], IMAGE_SIZE)
    assert result.level == 1
    assert result.score < 25


def test_many_large_defects_is_fatal():
    many = [detection("scratch", size=100, left=index * 100) for index in range(12)]
    result = grade(many, IMAGE_SIZE)
    assert result.level == 4


def test_low_confidence_is_ignored():
    result = grade([detection("fold", conf=0.1)], IMAGE_SIZE, conf_floor=0.25)
    assert result.level == 0


def test_veto_promotes_fold():
    """fold 严重度 4，面积虽小也应触发一票否决，至少 III 级。"""
    result = grade([detection("fold", size=100)], IMAGE_SIZE)
    assert result.level >= 3
    assert any("一票否决" in reason for reason in result.reasons)


def test_features_are_reported():
    result = grade([detection("porosity", size=80)], IMAGE_SIZE)
    assert result.features["count"] == 1
    assert 0 < result.features["area_ratio"] < 1
    assert result.features["severity"] == 3


def test_report_contains_level_and_reasons():
    result = grade([detection("oxidation", size=90)], IMAGE_SIZE)
    report = format_report(result, image_name="a.jpg", model_name="best.pt", elapsed_ms=12.3)
    assert "表面缺陷检测报告" in report
    assert result.level_name in report
    assert "判定依据" in report
    assert "12.3 ms" in report


@pytest.mark.parametrize("size", [40, 80, 160, 320])
def test_score_is_monotonic_in_area(size: int):
    small = grade([detection("scratch", size=size)], IMAGE_SIZE).score
    big = grade([detection("scratch", size=size * 2)], IMAGE_SIZE).score
    assert big >= small
