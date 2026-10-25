"""把标注画到原图上，用来肉眼检查标注质量。

训练前值得看一眼：框有没有跑偏、类别有没有标错、有没有漏标。
多边形标签会按外接框画（检测任务用的就是外接框）。

    python scripts/preview_dataset.py --n 4
    python scripts/preview_dataset.py --split test --n 6 --out docs/assets/test_preview.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_dataset, repo_path  # noqa: E402

# 6 个类别的颜色，尽量互相区分得开
COLORS = {
    0: (255, 214, 10),   # dark_spot  黄
    1: (255, 59, 48),    # fold       红
    2: (48, 209, 88),    # oxidation  绿
    3: (10, 132, 255),   # porosity   蓝
    4: (255, 92, 141),   # scratch    粉
    5: (191, 90, 242),   # stain      紫
}


def polygon_to_box(coords: list[float]) -> tuple[float, float, float, float]:
    xs, ys = coords[0::2], coords[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def read_boxes(label_path: Path) -> list[tuple[int, tuple[float, float, float, float]]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        boxes.append((int(parts[0]), polygon_to_box([float(v) for v in parts[1:]])))
    return boxes


def annotate(image: Image.Image, boxes, names: dict[int, str]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    line_width = max(2, width // 200)
    font = ImageFont.load_default(size=max(12, width // 32))
    for cls, (x0, y0, x1, y1) in boxes:
        color = COLORS.get(cls, (255, 255, 255))
        draw.rectangle([x0 * width, y0 * height, x1 * width, y1 * height], outline=color, width=line_width)
        label = names.get(cls, str(cls))
        text_box = draw.textbbox((0, 0), label, font=font)
        tw, th = text_box[2] - text_box[0], text_box[3] - text_box[1]
        tx, ty = x0 * width, max(0, y0 * height - th - 6)
        draw.rectangle([tx, ty, tx + tw + 6, ty + th + 6], fill=color)
        draw.text((tx + 3, ty + 2), label, fill=(0, 0, 0), font=font)
    return canvas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="可视化数据集标注")
    parser.add_argument("--data", default="configs/dataset.msdd.yaml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=4, help="拼几张图（2 列）")
    parser.add_argument("--out", default="docs/assets/dataset_preview.jpg")
    args = parser.parse_args(argv)

    cfg = load_dataset(args.data)
    names: dict[int, str] = cfg["names"]
    root = repo_path(cfg["path"])
    image_dir = root / "images" / args.split
    label_dir = root / "labels" / args.split

    images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if not images:
        raise SystemExit(f"{image_dir} 下没有图片")

    # 挑图：优先选有框的，并保证至少有一张正常样本（空标签）
    with_boxes = [p for p in images if (label_dir / f"{p.stem}.txt").stat().st_size > 0]
    empty = [p for p in images if (label_dir / f"{p.stem}.txt").stat().st_size == 0]
    chosen = []
    if empty:
        chosen.append(empty[len(empty) // 2])
    step = max(1, len(with_boxes) // max(1, args.n))
    chosen.extend(with_boxes[::step][: max(0, args.n - len(chosen))])
    chosen = chosen[: args.n]

    tiles = []
    for path in chosen:
        boxes = read_boxes(label_dir / f"{path.stem}.txt")
        tiles.append(annotate(Image.open(path), boxes, names))

    cols = 2
    rows = (len(tiles) + cols - 1) // cols
    tw, th = tiles[0].size
    gap = 8
    grid = Image.new("RGB", (cols * tw + (cols + 1) * gap, rows * th + (rows + 1) * gap), (24, 24, 24))
    for index, tile in enumerate(tiles):
        row, col = divmod(index, cols)
        grid.paste(tile, (gap + col * (tw + gap), gap + row * (th + gap)))

    target = repo_path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    grid.save(target, quality=92)
    print(f"已生成 {target}  ({len(tiles)} 张图，{grid.size[0]}x{grid.size[1]})")
    for path in chosen:
        count = len(read_boxes(label_dir / f"{path.stem}.txt"))
        print(f"  {path.name[:40]}...  框数={count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
