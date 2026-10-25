"""评估自研检测器（ResNet-FPN-CBAM）并写入和 YOLO 共用的结果格式。

    python -m src.evaluate_cnn --weights runs/cnn-cbam/best.pt --split test --tag cnn-cbam-test

YOLO 的结果由 `src.evaluate` 产出，这里补上另一条路线，两者最终汇到同一张对比表。
"""

from __future__ import annotations

import argparse
import sys

import torch
from torch.utils.data import DataLoader

from src.config import load_dataset, repo_path
from src.data.yolo_dataset import YoloDetectionDataset, collate_fn
from src.evaluate import save_result, write_table
from src.models.detector import build_detector
from src.train_cnn import evaluate_map


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.evaluate_cnn", description="评估自研检测器")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", default="configs/dataset.msdd.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    weights = repo_path(args.weights)
    if not weights.exists():
        raise SystemExit(f"找不到权重: {weights}")

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(str(weights), map_location="cpu", weights_only=False)
    meta = checkpoint.get("meta", {})
    names: dict[int, str] = meta.get("names") or load_dataset(args.dataset)["names"]

    model = build_detector(
        num_classes=meta.get("num_classes", len(names) + 1),
        backbone=meta.get("backbone", "resnet50"),
        use_cbam=meta.get("use_cbam", True),
        min_size=meta.get("imgsz", 512),
        max_size=meta.get("imgsz", 512),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    root = repo_path(load_dataset(args.dataset)["path"])
    dataset = YoloDetectionDataset(root, args.split)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        collate_fn=collate_fn, pin_memory=True,
    )
    print(f"评估 {args.split}: {len(dataset)} 张  device={device}")
    metrics = evaluate_map(model, loader, device)
    print(f"mAP50={metrics['mAP50']:.4f}  mAP50-95={metrics['mAP50-95']:.4f}  recall={metrics['recall']:.4f}")

    result = {
        "weights": str(weights),
        "size_mb": round(weights.stat().st_size / (1024 * 1024), 2),
        "params": int(sum(p.numel() for p in model.parameters())),
        "task": "detect",
        "split": args.split,
        "mAP50-95": metrics["mAP50-95"],
        "mAP50": metrics["mAP50"],
        "precision": None,
        "recall": metrics["recall"],
        "raw": metrics,
    }
    note = args.note or (
        f"{meta.get('backbone', 'resnet50')} + FPN {'+ CBAM' if meta.get('use_cbam') else ''}，"
        f"{args.split}"
    )
    path = save_result(result, args.tag, note, repo_path("results"))
    write_table(repo_path("results"))
    print(f"已写入 {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
