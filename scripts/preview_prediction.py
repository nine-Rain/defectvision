"""用训练好的模型跑几张图，把结果拼成一张预览图。

    python scripts/preview_prediction.py --weights runs/baseline/weights/best.pt
    python scripts/preview_prediction.py --weights runs/baseline-seg/weights/best.pt --out docs/assets/seg_prediction.jpg

分割模型画出来是掩码，检测模型画出来是框——README 里的效果图就是这么来的。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_dataset, repo_path  # noqa: E402


def pick_images(data: str, split: str, n: int) -> list[Path]:
    cfg = load_dataset(data)
    root = repo_path(cfg["path"])
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if not images:
        raise SystemExit(f"{image_dir} 下没有图片")
    # 优先挑有缺陷的图，不然预览里全是空图
    with_boxes = [p for p in images if (label_dir / f"{p.stem}.txt").stat().st_size > 0]
    pool = with_boxes or images
    step = max(1, len(pool) // n)
    return pool[::step][:n]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="模型预测可视化")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="configs/dataset.msdd.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", default="docs/assets/prediction.jpg")
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    weights = repo_path(args.weights)
    if not weights.exists():
        raise SystemExit(f"找不到权重: {weights}")

    model = YOLO(str(weights))
    images = pick_images(args.data, args.split, args.n)

    tiles = []
    for path in images:
        result = model.predict(str(path), conf=args.conf, verbose=False)[0]
        # plot() 返回 BGR，转成 RGB
        tiles.append(Image.fromarray(result.plot()[:, :, ::-1]))
        count = 0 if result.masks is None and result.boxes is None else len(result.boxes)
        has_mask = result.masks is not None
        print(f"  {path.name[:40]}...  目标={count}  掩码={'有' if has_mask else '无'}")

    cols = 2
    rows = (len(tiles) + cols - 1) // cols
    w, h = tiles[0].size
    gap = 8
    grid = Image.new("RGB", (cols * w + (cols + 1) * gap, rows * h + (rows + 1) * gap), (24, 24, 24))
    for index, tile in enumerate(tiles):
        row, col = divmod(index, cols)
        grid.paste(tile, (gap + col * (w + gap), gap + row * (h + gap)))

    target = repo_path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    grid.save(target, quality=92)
    print(f"已生成 {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
