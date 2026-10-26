"""从缺陷图的"干净区域"裁难负样本，专治"整体框"误检。

思路：模型会把整块零件表面当成缺陷（框住 80%~100% 面积）。这类误检的负样本
不是"没有零件的黑边"，而是"有零件、有纹理、但没有缺陷"的表面。
所以从训练集里有缺陷的图上，随机裁出**不与任何标注框相交**的区域，作为背景样本。

    python scripts/make_hard_negatives.py --limit 600          # 生成
    python scripts/make_hard_negatives.py --clean              # 删除已生成的

生成的文件带 hn_ 前缀，混进 train 目录（data/ 已被 .gitignore 忽略，可随时重生成）。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_dataset, repo_path  # noqa: E402


def read_boxes(label_path: Path) -> list[tuple[float, float, float, float]]:
    """读 YOLO 标签，返回归一化 (x0, y0, x1, y1)。"""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        coords = [float(v) for v in parts[1:]]
        xs, ys = coords[0::2], coords[1::2]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes


def overlaps(box, crop) -> float:
    """crop 与 box 的交集面积占 crop 的比例。"""
    ix0 = max(box[0], crop[0])
    iy0 = max(box[1], crop[1])
    ix1 = min(box[2], crop[2])
    iy1 = min(box[3], crop[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = (crop[2] - crop[0]) * (crop[3] - crop[1])
    return inter / area if area > 0 else 1.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成难负样本裁片")
    parser.add_argument("--dataset", default="configs/dataset.msdd.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--crop-min", type=int, default=320)
    parser.add_argument("--crop-max", type=int, default=512)
    parser.add_argument("--max-overlap", type=float, default=0.02, help="允许与标注框的最大重叠比例")
    parser.add_argument("--min-std", type=float, default=18.0, help="低于此值视为纯色区域，跳过")
    parser.add_argument("--prefix", default="hn_")
    parser.add_argument(
        "--augment-background",
        type=int,
        default=0,
        help="额外把已有的纯背景图做 N 份旋转/翻转增广（直接对应'整件无缺陷'这种负样本）",
    )
    parser.add_argument("--clean", action="store_true", help="删除已生成的难负样本")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    cfg = load_dataset(args.dataset)
    root = repo_path(cfg["path"])
    image_dir = root / "images" / args.split
    label_dir = root / "labels" / args.split

    existing = sorted(image_dir.glob(f"{args.prefix}*.jpg"))
    if args.clean:
        removed = 0
        for path in existing:
            path.unlink()
            (label_dir / f"{path.stem}.txt").unlink(missing_ok=True)
            removed += 1
        print(f"已删除 {removed} 组难负样本")
        return 0

    if existing:
        print(f"已存在 {len(existing)} 组难负样本；如需重新生成请先 --clean")
        return 0

    random.seed(args.seed)
    all_images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".png"})
    # 只从"有缺陷的图"上裁，纯背景图本来就没缺陷，裁了意义不大
    images = [p for p in all_images if read_boxes(label_dir / f"{p.stem}.txt")]
    random.shuffle(images)

    made = 0
    tried = 0
    for image_path in images:
        if made >= args.limit:
            break
        boxes = read_boxes(label_dir / f"{image_path.stem}.txt")
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]

        for _ in range(6):  # 每张图最多试几个位置
            tried += 1
            size = random.randint(args.crop_min, args.crop_max)
            size = min(size, width, height)
            x0 = random.randint(0, width - size)
            y0 = random.randint(0, height - size)
            crop_norm = (x0 / width, y0 / height, (x0 + size) / width, (y0 + size) / height)
            if any(overlaps(box, crop_norm) > args.max_overlap for box in boxes):
                continue

            patch = image[y0 : y0 + size, x0 : x0 + size]
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            if float(gray.std()) < args.min_std:
                continue  # 纯色/黑边

            name = f"{args.prefix}{made:04d}"
            cv2.imwrite(str(image_dir / f"{name}.jpg"), patch, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (label_dir / f"{name}.txt").write_text("", encoding="utf-8")  # 空标签=背景
            made += 1
            break

    print(f"尝试 {tried} 次，生成 {made} 张难负样本（尺寸 {args.crop_min}~{args.crop_max}）")

    # 纯背景图增广：缺陷图的干净区域往往不够大，而"整件无缺陷"正是模型误画整体框的场景。
    # 圆形零件做 90/180/270 度旋转与翻转是自然增广，不改变语义。
    if args.augment_background > 0:
        background = [
            p for p in all_images if not read_boxes(label_dir / f"{p.stem}.txt")
        ]
        added = 0
        for image_path in background:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            variants = [
                cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
                cv2.rotate(image, cv2.ROTATE_180),
                cv2.flip(image, 1),
            ]
            for variant in variants[: args.augment_background]:
                name = f"{args.prefix}bg_{added:04d}"
                cv2.imwrite(str(image_dir / f"{name}.jpg"), variant, [cv2.IMWRITE_JPEG_QUALITY, 95])
                (label_dir / f"{name}.txt").write_text("", encoding="utf-8")
                added += 1
        print(f"背景图增广 {len(background)} 张 -> 新增 {added} 张")
    print(f"位置: {image_dir}（{args.prefix}*.jpg，空标签）")
    print("下一步：python -m src.train --model yolo11n.pt --epochs 50 --name yolo11n-hn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
