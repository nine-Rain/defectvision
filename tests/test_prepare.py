"""数据体检与切分的测试。用临时目录造小数据集，不依赖真实数据。"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.config import load_dataset
from src.prepare import (
    find_duplicates,
    parse_label_file,
    split_dataset,
    validate_dataset,
)

NAMES = {0: "scratch", 1: "dent"}


def make_image(root: Path, split: str, stem: str, label: str, color: int = 0) -> None:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), (color, color, color)).save(image_dir / f"{stem}.jpg")
    (label_dir / f"{stem}.txt").write_text(label, encoding="utf-8")


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    make_image(tmp_path, "train", "a", "0 0.5 0.5 0.2 0.2\n", color=10)
    make_image(tmp_path, "train", "b", "1 0.4 0.4 0.1 0.1\n0 0.6 0.6 0.1 0.1\n", color=20)
    make_image(tmp_path, "val", "c", "0 0.5 0.5 0.3 0.3\n", color=30)
    return tmp_path


def test_valid_dataset_has_no_errors(dataset: Path):
    stats, errors = validate_dataset(dataset, NAMES)
    assert errors == []
    assert stats["train"].images == 2
    assert stats["train"].boxes == 3
    assert stats["train"].per_class["scratch"] == 2
    assert stats["val"].boxes == 1


def test_empty_label_counts_as_negative(dataset: Path):
    (dataset / "labels" / "train" / "b.txt").write_text("", encoding="utf-8")
    stats, errors = validate_dataset(dataset, NAMES)
    assert errors == []
    assert stats["train"].empty_labels == 1


def test_missing_label_is_reported(dataset: Path):
    (dataset / "labels" / "val" / "c.txt").unlink()
    stats, errors = validate_dataset(dataset, NAMES)
    assert stats["val"].missing_labels == 1
    assert any("缺标签文件" in e for e in errors)


def test_class_out_of_range(dataset: Path):
    (dataset / "labels" / "train" / "a.txt").write_text("7 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    _, errors = validate_dataset(dataset, NAMES)
    assert any("超出范围" in e for e in errors)


def test_even_field_count_is_rejected(tmp_path: Path):
    # 6 个字段既不是检测(5)也不是分割(1+2n)，一定是写错了
    path = tmp_path / "x.txt"
    path.write_text("0 0.1 0.1 0.2 0.2 0.3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="奇数"):
        parse_label_file(path, num_classes=2)


def test_out_of_range_coordinates(tmp_path: Path):
    path = tmp_path / "x.txt"
    path.write_text("0 1.5 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="0~1"):
        parse_label_file(path, num_classes=2)


def test_segmentation_polygon_format_is_accepted(tmp_path: Path):
    path = tmp_path / "x.txt"
    path.write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8")
    boxes = parse_label_file(path, num_classes=2)
    assert boxes[0][0] == 0
    assert len(boxes[0][1]) == 6


def test_duplicates_across_splits_are_found(dataset: Path):
    # 复制同一张图到 val，模拟训练集泄漏
    source = dataset / "images" / "train" / "a.jpg"
    (dataset / "images" / "val" / "leak.jpg").write_bytes(source.read_bytes())
    duplicates = find_duplicates(dataset)
    assert len(duplicates) == 1
    paths = next(iter(duplicates.values()))
    splits = {p.parent.name for p in paths}
    assert splits == {"train", "val"}


def test_split_dataset(tmp_path: Path):
    src = tmp_path / "raw"
    for index in range(10):
        make_image(src, "flat", f"img{index}", "0 0.5 0.5 0.2 0.2\n", color=index)
    # split_dataset 直接扫 images/ 下的所有图，不关心是否分 split 子目录
    flat_images = src / "images" / "flat"
    flat_labels = src / "labels" / "flat"
    for image in flat_images.glob("*.jpg"):
        image.rename(src / "images" / image.name)
    for label in flat_labels.glob("*.txt"):
        label.rename(src / "labels" / label.name)
    flat_images.rmdir()
    flat_labels.rmdir()

    out = tmp_path / "dataset"
    counts = split_dataset(src, out, NAMES, ratios=(0.8, 0.1, 0.1), seed=0)
    assert counts["train"] == 8
    assert counts["val"] == 1
    assert counts["test"] == 1
    assert (out / "dataset.yaml").exists()
    loaded = load_dataset(out / "dataset.yaml")
    assert loaded["names"] == NAMES
    assert len(list((out / "images" / "val").glob("*.jpg"))) == 1
    # 每张图都应该有对应的（可能为空的）标签文件
    for image in (out / "images" / "train").glob("*.jpg"):
        assert (out / "labels" / "train" / f"{image.stem}.txt").exists()
