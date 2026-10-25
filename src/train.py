"""训练入口：把 configs/train.yaml 翻译成一次 Ultralytics 训练。

    python -m src.train                # 按配置训练
    python -m src.train --smoke        # 1 个 epoch 的冒烟测试，先确认流水线通
    python -m src.train --dry-run      # 只打印将要执行的参数，不训练
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.config import TrainConfig, load_dataset, repo_path


def build_config(args: argparse.Namespace) -> TrainConfig:
    data: dict[str, Any] = TrainConfig.from_yaml(args.config).to_dict()
    overrides = {
        "model": args.model,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "name": args.name,
    }
    for key, value in overrides.items():
        if value is not None:
            data[key] = value
    if args.smoke:
        data["epochs"] = 1
        data["imgsz"] = 320
        data["name"] = "smoke"
        if data["batch"] == -1:
            data["batch"] = 4
        else:
            data["batch"] = min(int(data["batch"]), 4)
    return TrainConfig(**data)


def check_dataset(dataset_config: str) -> None:
    cfg = load_dataset(dataset_config)
    root = repo_path(cfg["path"])
    train_images = root / "images" / "train"
    has_images = train_images.is_dir() and any(
        p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for p in train_images.iterdir()
        if p.is_file()
    )
    if not has_images:
        raise SystemExit(
            f"训练集为空或不存在: {train_images}\n"
            "先跑 python scripts/make_demo_dataset.py 生成示例数据，"
            "或者准备自己的数据后改 configs/dataset.yaml 的 path。"
        )


def train_kwargs(cfg: TrainConfig) -> dict[str, Any]:
    return {
        "data": str(repo_path(cfg.dataset)),
        "epochs": cfg.epochs,
        "imgsz": cfg.imgsz,
        "batch": cfg.batch,
        "patience": cfg.patience,
        "seed": cfg.seed,
        "device": cfg.resolve_device(),
        "workers": cfg.workers,
        "project": str(repo_path(cfg.project)),
        "name": cfg.name,
        "exist_ok": True,
        "plots": True,
        "val": True,
        "verbose": True,
    }


def run(args: argparse.Namespace) -> int:
    cfg = build_config(args)
    check_dataset(cfg.dataset)
    kwargs = train_kwargs(cfg)

    print("训练配置")
    for key, value in kwargs.items():
        print(f"  {key:<10}{value}")
    if args.dry_run:
        print("\n--dry-run：以上参数未执行。")
        return 0

    from ultralytics import YOLO  # 延迟导入，--dry-run 时不需要装 ultralytics

    if args.resume:
        model = YOLO(str(repo_path(args.resume)))
        results = model.train(resume=True)
    else:
        model = YOLO(cfg.model)
        results = model.train(**kwargs)

    save_dir = Path(getattr(model.trainer, "save_dir", repo_path(cfg.project) / cfg.name))
    summary = {
        "config": asdict(cfg),
        "save_dir": str(save_dir),
        "metrics": getattr(results, "results_dict", {}) or {},
        "speed_ms": getattr(results, "speed", {}) or {},
    }
    (save_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n权重目录: {save_dir / 'weights'}")
    print(f"下一步: python -m src.evaluate --weights {save_dir / 'weights' / 'best.pt'} --tag {cfg.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.train", description="训练检测/分割模型")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model", help="如 yolo11n.pt / yolo11s.pt / yolo11n-seg.pt")
    parser.add_argument("--dataset", help="数据集配置，默认用 train.yaml 里的 dataset")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int, help="-1 表示按显存自动")
    parser.add_argument("--device", help="auto | cpu | 0 | 0,1")
    parser.add_argument("--name", help="实验名，决定 runs/<name>/")
    parser.add_argument("--resume", help="从某个 last.pt 续训")
    parser.add_argument("--smoke", action="store_true", help="1 epoch + 320 尺寸的冒烟训练")
    parser.add_argument("--dry-run", action="store_true", help="只打印参数")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
