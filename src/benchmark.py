"""延迟与体积基准。

面试里"我部署到了端侧"必须带数字：p50/p95 延迟、模型体积、精度损失。
只有均值没有 p95，说明没做过真实部署。

    python -m src.benchmark --weights models/best.onnx --runs 30
    python -m src.benchmark --weights runs/baseline/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from src.config import load_dataset, repo_path


def collect_images(data: str, split: str, limit: int) -> list[Path]:
    cfg = load_dataset(data)
    root = repo_path(cfg["path"])
    image_dir = root / "images" / split
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in sorted(image_dir.glob("*")) if p.suffix.lower() in suffixes]
    if not images:
        raise SystemExit(f"在 {image_dir} 下没找到图片，先准备数据")
    return images[:limit]


def benchmark(
    weights: Path, images: list[Path], warmup: int = 5, runs: int = 30, imgsz: int = 640
) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    for image in images[:warmup]:
        model.predict(str(image), imgsz=imgsz, verbose=False)

    samples = []
    for index in range(runs):
        image = images[index % len(images)]
        start = time.perf_counter()
        model.predict(str(image), imgsz=imgsz, verbose=False)
        samples.append((time.perf_counter() - start) * 1000)

    samples.sort()
    params = None
    model_obj = getattr(model, "model", None)
    if model_obj is not None:
        try:
            params = int(sum(p.numel() for p in model_obj.parameters()))
        except Exception:  # noqa: BLE001
            params = None

    return {
        "weights": str(weights),
        "backend": weights.suffix.lstrip("."),
        "imgsz": imgsz,
        "runs": runs,
        "size_mb": round(weights.stat().st_size / (1024 * 1024), 2),
        "params": params,
        "mean_ms": round(statistics.fmean(samples), 2),
        "p50_ms": round(statistics.median(samples), 2),
        "p95_ms": round(samples[min(int(len(samples) * 0.95), len(samples) - 1)], 2),
        "min_ms": round(samples[0], 2),
        "max_ms": round(samples[-1], 2),
        "fps": round(1000 / statistics.fmean(samples), 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.benchmark", description="延迟基准")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args(argv)

    weights = repo_path(args.weights)
    if not weights.exists():
        raise SystemExit(f"找不到权重: {weights}")

    images = collect_images(args.data, args.split, limit=max(args.runs, args.warmup, 1))
    result = benchmark(weights, images, args.warmup, args.runs, args.imgsz)

    results_dir = repo_path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / f"benchmark_{weights.stem}.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{weights.name}  ({result['backend']}, imgsz={args.imgsz}, n={args.runs})")
    print(f"  体积      {result['size_mb']} MB")
    print(f"  平均       {result['mean_ms']} ms")
    print(f"  p50 / p95  {result['p50_ms']} / {result['p95_ms']} ms")
    print(f"  吞吐       {result['fps']} FPS")
    print(f"已写入 {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
