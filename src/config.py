"""配置加载与校验。

原则：配置写错要在开训前 0.1 秒报错，而不是等训练跑到第 3 个 epoch 才崩。
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_path(*parts: str | Path) -> Path:
    """相对仓库根目录解析路径，避免受当前工作目录影响。"""
    path = Path(*parts)
    return path if path.is_absolute() else REPO_ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置，路径相对仓库根目录。"""
    resolved = repo_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"配置文件不存在: {resolved}")
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是映射（key: value）: {resolved}")
    return data


def load_dataset(path: str | Path = "configs/dataset.yaml") -> dict[str, Any]:
    """读取数据集配置，并把 names 统一成 {int: str}。"""
    data = load_yaml(path)
    names = data.get("names")
    if isinstance(names, list):
        data["names"] = {i: str(n) for i, n in enumerate(names)}
    elif isinstance(names, dict):
        try:
            data["names"] = {int(k): str(v) for k, v in names.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("dataset.yaml 里的 names 必须是 整数->类别名 的映射") from exc
    else:
        raise ValueError("dataset.yaml 缺少 names 字段")
    if not data["names"]:
        raise ValueError("dataset.yaml 的 names 不能为空")
    if sorted(data["names"]) != list(range(len(data["names"]))):
        raise ValueError("names 的类别 id 必须是 0..N-1 连续整数")
    return data


def materialize_dataset(
    path: str | Path = "configs/dataset.yaml", out: str | Path | None = None
) -> Path:
    """把数据集配置里的相对 path 转成绝对路径，落到 runs/_resolved_dataset.yaml。

    Ultralytics 解析 dataset.yaml 时，相对 path 是拼到它自己的 settings.datasets_dir
    上的，不是相对配置文件——直接传原始 yaml 会出现"在 projects/datasets 下面找图片"
    这种莫名其妙的报错。这里统一解析好再交给它。
    """
    cfg = load_dataset(path)
    cfg["path"] = str(repo_path(cfg["path"]).resolve())
    target = repo_path(out) if out else repo_path("runs") / "_resolved_dataset.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return target


@dataclass(slots=True)
class TrainConfig:
    """训练超参。字段名与 Ultralytics 的参数名保持一致，便于直接透传。"""

    model: str = "yolo11n.pt"
    dataset: str = "configs/dataset.yaml"
    epochs: int = 80
    imgsz: int = 640
    batch: int = 16
    patience: int = 20
    seed: int = 42
    device: str = "auto"
    workers: int = 4
    project: str = "runs"
    name: str = "baseline"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not str(self.model).strip():
            raise ValueError("model 不能为空，例如 yolo11n.pt / yolo11n-seg.pt")
        if self.epochs < 1:
            raise ValueError(f"epochs 必须 >= 1，当前 {self.epochs}")
        if self.imgsz < 32 or self.imgsz % 32 != 0:
            raise ValueError(f"imgsz 必须是 >=32 的 32 倍数，当前 {self.imgsz}")
        if self.batch == 0 or self.batch < -1:
            raise ValueError(f"batch 必须 >= 1 或等于 -1（自动），当前 {self.batch}")
        if self.patience < 0:
            raise ValueError(f"patience 必须 >= 0，当前 {self.patience}")
        if self.workers < 0:
            raise ValueError(f"workers 必须 >= 0，当前 {self.workers}")
        if not str(self.name).strip():
            raise ValueError("name 不能为空，它决定 runs/ 下的实验目录名")

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/train.yaml") -> TrainConfig:
        raw = load_yaml(path)
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(
                f"train.yaml 里有未知字段: {', '.join(unknown)}；"
                f"可用字段: {', '.join(sorted(known))}"
            )
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolve_device(self) -> str:
        """auto -> 有 GPU 用 0，Apple 芯片用 mps，否则 cpu。"""
        if self.device != "auto":
            return self.device
        if importlib.util.find_spec("torch") is None:
            return "cpu"
        import torch  # 延迟导入：不训练时不需要 torch

        if torch.cuda.is_available():
            return "0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
