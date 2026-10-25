"""生成一个合成数据集，让你在 10 分钟内跑通全流程。

它不是真实数据，训练出来的模型也毫无业务价值——唯一的作用是：
在你标注第一张真实图片之前，先证明「训练→评估→导出→部署」这条链路是通的。

用法：
    python scripts/make_demo_dataset.py --out data/demo --n 160
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

from src.config import repo_path  # noqa: E402

NAMES = {0: "scratch", 1: "dent", 2: "stain"}
WIDTH, HEIGHT = 640, 480


def _background(rng: random.Random) -> Image.Image:
    """金属/塑料质感的底色，带一点渐变和噪点。"""
    base = rng.randint(110, 190)
    image = Image.new("RGB", (WIDTH, HEIGHT), (base, base + rng.randint(-8, 8), base + rng.randint(-8, 8)))
    draw = ImageDraw.Draw(image)
    for y in range(HEIGHT):
        shade = int(18 * math.sin(y / HEIGHT * math.pi))
        draw.line([(0, y), (WIDTH, y)], fill=(base + shade, base + shade, base + shade))
    noise = Image.effect_noise((WIDTH, HEIGHT), 12).convert("L")
    return Image.blend(image, Image.merge("RGB", (noise, noise, noise)), 0.08)


def _scratch(draw: ImageDraw.ImageDraw, rng: random.Random) -> tuple[float, float, float, float]:
    x0, y0 = rng.uniform(60, WIDTH - 60), rng.uniform(60, HEIGHT - 60)
    angle = rng.uniform(0, math.pi)
    length = rng.uniform(50, 140)
    x1 = x0 + length * math.cos(angle)
    y1 = y0 + length * math.sin(angle)
    width = rng.randint(1, 3)
    draw.line([(x0, y0), (x1, y1)], fill=(rng.randint(30, 70),) * 3, width=width)
    pad = 4
    return (
        min(x0, x1) - pad,
        min(y0, y1) - pad,
        max(x0, x1) + pad,
        max(y0, y1) + pad,
    )


def _dent(draw: ImageDraw.ImageDraw, rng: random.Random) -> tuple[float, float, float, float]:
    cx, cy = rng.uniform(80, WIDTH - 80), rng.uniform(80, HEIGHT - 80)
    rx, ry = rng.uniform(18, 45), rng.uniform(18, 45)
    box = [cx - rx, cy - ry, cx + rx, cy + ry]
    draw.ellipse(box, fill=(rng.randint(60, 95),) * 3, outline=(40, 40, 40), width=2)
    return tuple(box)  # type: ignore[return-value]


def _stain(draw: ImageDraw.ImageDraw, rng: random.Random) -> tuple[float, float, float, float]:
    cx, cy = rng.uniform(80, WIDTH - 80), rng.uniform(80, HEIGHT - 80)
    rx, ry = rng.uniform(25, 60), rng.uniform(25, 60)
    color = (rng.randint(120, 165), rng.randint(95, 130), rng.randint(40, 80))
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=color)
    return (cx - rx, cy - ry, cx + rx, cy + ry)


GENERATORS = {0: _scratch, 1: _dent, 2: _stain}


def make_image(rng: random.Random) -> tuple[Image.Image, list[tuple[int, tuple[float, ...]]]]:
    canvas = _background(rng)
    draw = ImageDraw.Draw(canvas)
    labels = []
    # 约 10% 是负样本（整张图没有缺陷）——真实产线里也大量存在
    count = 0 if rng.random() < 0.1 else rng.randint(1, 3)
    for _ in range(count):
        cls = rng.choices([0, 1, 2], weights=[0.4, 0.3, 0.3])[0]
        box = GENERATORS[cls](draw, rng)
        labels.append((cls, box))
    canvas = canvas.filter(ImageFilter.GaussianBlur(0.4))
    return canvas, labels


def to_yolo(box: tuple[float, ...]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    x0, x1 = sorted((max(0.0, x0), min(WIDTH - 1.0, x1)))
    y0, y1 = sorted((max(0.0, y0), min(HEIGHT - 1.0, y1)))
    cx, cy = (x0 + x1) / 2 / WIDTH, (y0 + y1) / 2 / HEIGHT
    w, h = (x1 - x0) / WIDTH, (y1 - y0) / HEIGHT
    return cx, cy, w, h


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成合成缺陷数据集")
    parser.add_argument("--out", default="data/demo")
    parser.add_argument("--n", type=int, default=160)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    root = repo_path(args.out)
    rng = random.Random(args.seed)
    n_val = max(1, int(args.n * args.val_ratio))

    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for index in range(args.n):
        split = "val" if index < n_val else "train"
        image, labels = make_image(rng)
        stem = f"{split}_{index:04d}"
        image.save(root / "images" / split / f"{stem}.jpg", quality=92)
        lines = []
        for cls, box in labels:
            cx, cy, w, h = to_yolo(box)
            if w <= 0 or h <= 0:
                continue
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (root / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    import yaml

    (root / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "names": NAMES,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    print(f"已生成 {args.n} 张合成图 -> {root}")
    print("下一步（1 epoch 冒烟训练，CPU 也行）:")
    print("  python -m src.train --smoke --dataset configs/dataset.demo.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
