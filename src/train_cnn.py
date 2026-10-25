"""训练改进版检测器（ResNet + FPN + CBAM），并用 mAP 评估。

    # 完整模型
    python -m src.train_cnn --name cnn-cbam --epochs 20

    # 消融：去掉 CBAM
    python -m src.train_cnn --name cnn-nocbam --no-cbam --epochs 20

    # 冒烟
    python -m src.train_cnn --smoke

训练日志里每个 epoch 打一行，结束后把指标写进 results/<name>.json，
和 YOLO 的结果共用同一张对比表。
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import load_dataset, repo_path
from src.data.yolo_dataset import YoloDetectionDataset, collate_fn
from src.evaluate import save_result, write_table
from src.models.detector import build_detector, count_parameters


def build_loader(
    root: Path, split: str, batch: int, workers: int, shuffle: bool, augment: bool = False
) -> DataLoader:
    dataset = YoloDetectionDataset(root, split, augment=augment)
    if len(dataset) == 0:
        raise SystemExit(f"{split} 集是空的: {root / 'images' / split}")
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


@torch.no_grad()
def evaluate_map(model, loader, device: str) -> dict[str, float]:
    from torchmetrics.detection import MeanAveragePrecision

    # 默认 backend 是 pycocotools，它没有现成 wheel，Windows 上装起来很麻烦；
    # faster-coco-eval 是纯 C++ 实现、有预编译轮子，结果一致且更快。
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="faster_coco_eval")
    model.eval()
    for images, targets in loader:
        outputs = model([image.to(device) for image in images])
        metric.update(
            [{k: v.detach().cpu() for k, v in out.items()} for out in outputs],
            [
                {k: v.cpu() if torch.is_tensor(v) else v for k, v in target.items()}
                for target in targets
            ],
        )
    result = metric.compute()
    return {
        "mAP50-95": float(result["map"]),
        "mAP50": float(result["map_50"]),
        "mAP75": float(result["map_75"]),
        "recall": float(result["mar_100"]),
    }


def train(args: argparse.Namespace) -> int:
    cfg = load_dataset(args.dataset)
    names: dict[int, str] = cfg["names"]
    root = repo_path(cfg["path"])
    num_classes = len(names) + 1  # 0 号是背景

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    epochs = 1 if args.smoke else args.epochs
    imgsz = 320 if args.smoke else args.imgsz
    name = "smoke-cnn" if args.smoke else args.name

    print(f"数据集: {root}  ({len(names)} 类)")
    print(f"设备: {device}   epochs={epochs}  imgsz={imgsz}  batch={args.batch}")
    print(f"骨干: {args.backbone}   FPN: 是   CBAM: {'是' if args.cbam else '否'}")

    train_loader = build_loader(
        root, "train", args.batch, args.workers, shuffle=True, augment=args.augment
    )
    val_loader = build_loader(root, "val", args.batch, args.workers, shuffle=False)

    model = build_detector(
        num_classes=num_classes,
        backbone=args.backbone,
        use_cbam=args.cbam,
        min_size=imgsz,
        max_size=imgsz,
        pretrained=True,
    )
    model.to(device)
    print(f"可训练参数: {count_parameters(model) / 1e6:.1f} M")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.lr * 0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device == "cuda")

    save_dir = repo_path("runs") / name
    save_dir.mkdir(parents=True, exist_ok=True)
    history_path = save_dir / "history.csv"
    history = [["epoch", "loss", "mAP50", "mAP50-95", "recall", "minutes"]]

    best_map = -1.0
    best_metrics: dict[str, float] = {}
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        running = 0.0
        for step, (images, targets) in enumerate(train_loader, start=1):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device == "cuda"):
                loss_dict = model(images, targets)
                loss = sum(value for value in loss_dict.values())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())

            if step % 50 == 0:
                print(f"  epoch {epoch}  step {step}/{len(train_loader)}  loss={running / step:.3f}", flush=True)

        scheduler.step()
        metrics = evaluate_map(model, val_loader, device)
        minutes = (time.time() - epoch_start) / 60
        print(
            f"[epoch {epoch}/{epochs}] loss={running / max(1, step):.3f}  "
            f"mAP50={metrics['mAP50']:.4f}  mAP50-95={metrics['mAP50-95']:.4f}  "
            f"recall={metrics['recall']:.4f}  ({minutes:.1f} min)",
            flush=True,
        )
        history.append(
            [epoch, f"{running / max(1, step):.4f}", f"{metrics['mAP50']:.4f}",
             f"{metrics['mAP50-95']:.4f}", f"{metrics['recall']:.4f}", f"{minutes:.2f}"]
        )
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows(history)

        if metrics["mAP50-95"] > best_map:
            best_map = metrics["mAP50-95"]
            best_metrics = metrics
            torch.save(
                {
                    "model": model.state_dict(),
                    "meta": {
                        "names": names,
                        "num_classes": num_classes,
                        "backbone": args.backbone,
                        "use_cbam": args.cbam,
                        "imgsz": imgsz,
                    },
                },
                save_dir / "best.pt",
            )

    elapsed = (time.time() - started) / 60
    print(f"\n训练结束，用时 {elapsed:.1f} 分钟。最优 mAP50-95={best_map:.4f}")
    print(f"权重: {save_dir / 'best.pt'}")

    # 写进和 YOLO 共用的结果格式，这样对比表会自动带上这一行
    result = {
        "weights": str(save_dir / "best.pt"),
        "size_mb": round((save_dir / "best.pt").stat().st_size / (1024 * 1024), 2),
        "params": count_parameters(model),
        "task": "detect",
        "split": "val",
        "mAP50-95": best_metrics.get("mAP50-95"),
        "mAP50": best_metrics.get("mAP50"),
        "precision": None,
        "recall": best_metrics.get("recall"),
        "raw": best_metrics,
    }
    note = f"{args.backbone} + FPN {'+ CBAM' if args.cbam else ''}，{epochs}ep @{imgsz}"
    path = save_result(result, name, note, repo_path("results"))
    write_table(repo_path("results"))
    print(f"指标已写入 {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.train_cnn", description="训练改进检测器")
    parser.add_argument("--dataset", default="configs/dataset.msdd.yaml")
    parser.add_argument("--backbone", default="resnet50", choices=["resnet34", "resnet50"])
    parser.add_argument("--cbam", dest="cbam", action="store_true", default=True)
    parser.add_argument("--no-cbam", dest="cbam", action="store_false")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--name", default="cnn-cbam")
    parser.add_argument("--no-amp", dest="amp", action="store_false", default=True)
    parser.add_argument(
        "--augment",
        action="store_true",
        help="训练集开水平翻转 + 色彩扰动（YOLO 路线一直开着，CNN 路线默认关以便复现旧结果）",
    )
    parser.add_argument("--smoke", action="store_true", help="1 epoch + 320 尺寸")
    args = parser.parse_args(argv)
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
