"""配置加载与校验的测试。这些用例不需要装 torch / ultralytics。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import (
    REPO_ROOT,
    TrainConfig,
    load_dataset,
    load_yaml,
    materialize_dataset,
    repo_path,
)


def test_train_config_reads_shipped_yaml():
    cfg = TrainConfig.from_yaml("configs/train.yaml")
    assert cfg.epochs >= 1
    assert cfg.imgsz % 32 == 0
    assert cfg.batch != 0
    # 没装 torch 时必须回落到 cpu，而不是抛 ImportError
    assert cfg.resolve_device() in {"cpu", "0", "mps"}


def test_unknown_field_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("modle: yolo11n.pt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="未知字段"):
        TrainConfig.from_yaml(bad)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"imgsz": 100},
        {"batch": 0},
        {"patience": -1},
        {"name": "  "},
    ],
)
def test_invalid_values_raise(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


def test_dataset_yaml_names_are_int_keyed():
    cfg = load_dataset("configs/dataset.yaml")
    assert cfg["names"] == {0: "scratch", 1: "dent", 2: "stain"}


def test_dataset_yaml_missing_names(tmp_path):
    bad = tmp_path / "d.yaml"
    bad.write_text("path: data/x\ntrain: images/train\nval: images/val\n", encoding="utf-8")
    with pytest.raises(ValueError, match="names"):
        load_dataset(bad)


def test_load_yaml_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "nope.yaml")


def test_repo_path_is_absolute():
    assert repo_path("configs").is_absolute()
    assert repo_path("configs").parent == REPO_ROOT


def test_materialize_dataset_absolutizes_path(tmp_path):
    """相对 path 必须被改写成绝对路径，否则 Ultralytics 会拼到自己的 datasets_dir 上。"""
    source = tmp_path / "d.yaml"
    source.write_text(
        "path: data/msdd\ntrain: images/train\nval: images/val\nnames:\n  0: a\n",
        encoding="utf-8",
    )
    output = materialize_dataset(source, out=tmp_path / "resolved.yaml")
    data = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert Path(data["path"]).is_absolute()
    assert data["path"].endswith("data\\msdd") or data["path"].endswith("data/msdd")
    assert data["names"] == {0: "a"}
