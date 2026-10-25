"""把"自己拍的照片 + 标注"整理成项目能直接训练的 YOLO 数据集。

对应开题里"自建缺陷数据集"那一项：手机拍几十张、用 Label Studio 标完，
一条命令就能接进现有训练/评估/判级流程，不用手写转换脚本。

支持两种输入：

    # 1. 已经是 YOLO 格式（图片旁边有同名 .txt）
    python scripts/import_custom_dataset.py --src D:/photos --format yolo \
        --names scratch,stain --out data/custom

    # 2. Label Studio 导出的 JSON（矩形框）
    python scripts/import_custom_dataset.py --src D:/photos --format labelstudio \
        --annotations D:/photos/export.json --out data/custom

输出结构（和 MSDD / NEU-DET 一致）：

    data/custom/images/{train,val,test}/
    data/custom/labels/{train,val,test}/
    configs/dataset.custom.yaml

没有标注的图片会生成空标签文件，作为**负样本**——这在工业质检里非常重要，
全是缺陷样本会让模型到处误检。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import repo_path  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(src: Path) -> list[Path]:
    return sorted(p for p in src.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def parse_labelstudio(json_path: Path) -> tuple[dict[str, list[tuple[int, float, float, float, float]]], list[str]]:
    """解析 Label Studio 的矩形框导出。

    返回 {图片文件名: [(class_id, cx, cy, w, h), ...]} 和类别名列表。
    Label Studio 的坐标是**百分比**（0~100），x/y 是左上角，需要转成归一化中心点格式。
    """
    tasks = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise SystemExit("Label Studio 导出应该是 JSON 数组")

    # 类别按出现顺序编号，保证 id 稳定
    classes: list[str] = []
    boxes: dict[str, list[tuple[int, float, float, float, float]]] = {}

    for task in tasks:
        image_name = Path(task.get("data", {}).get("image", "")).name
        if not image_name:
            continue
        entries = boxes.setdefault(image_name, [])
        for annotation in task.get("annotations", []):
            for result in annotation.get("result", []):
                value = result.get("value", {})
                labels = value.get("rectanglelabels") or []
                if not labels:
                    continue  # 多边形/关键点暂不支持，跳过而不是报错
                label = labels[0]
                if label not in classes:
                    classes.append(label)
                x = float(value.get("x", 0)) / 100.0
                y = float(value.get("y", 0)) / 100.0
                w = float(value.get("width", 0)) / 100.0
                h = float(value.get("height", 0)) / 100.0
                if w <= 0 or h <= 0:
                    continue
                entries.append((classes.index(label), x + w / 2, y + h / 2, w, h))
    return boxes, classes


def read_yolo_labels(image: Path) -> list[tuple[int, float, float, float, float]]:
    label = image.with_suffix(".txt")
    if not label.exists():
        return []
    rows = []
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            rows.append((int(parts[0]), *(float(v) for v in parts[1:])))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把自建数据整理成 YOLO 数据集")
    parser.add_argument("--src", required=True, help="图片目录（可递归）")
    parser.add_argument("--format", choices=["yolo", "labelstudio"], default="yolo")
    parser.add_argument("--annotations", help="labelstudio 模式下的 JSON 路径")
    parser.add_argument("--names", help="逗号分隔的类别名（yolo 模式必填；labelstudio 模式自动读取）")
    parser.add_argument("--out", default="data/custom")
    parser.add_argument("--slug", default="custom")
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--move", action="store_true", help="移动原图而不是复制")
    args = parser.parse_args(argv)

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"源目录不存在: {src}")
    images = list_images(src)
    if not images:
        raise SystemExit(f"{src} 下没有图片")

    if args.format == "labelstudio":
        if not args.annotations:
            raise SystemExit("labelstudio 模式需要 --annotations")
        boxes_map, classes = parse_labelstudio(Path(args.annotations))
        if not classes:
            raise SystemExit("标注文件里没有矩形框，检查是否用了多边形标注")
        labels_for = lambda image: boxes_map.get(image.name, [])  # noqa: E731
    else:
        if not args.names:
            raise SystemExit("yolo 模式需要 --names，例如 --names scratch,stain")
        classes = [name.strip() for name in args.names.split(",") if name.strip()]
        labels_for = read_yolo_labels  # noqa: F811

    print(f"图片 {len(images)} 张，类别 {len(classes)} 个: {classes}")

    out_root = repo_path(args.out)
    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    shuffled = list(images)
    rng.shuffle(shuffled)
    total = len(shuffled)
    n_train = int(total * args.ratios[0])
    n_val = int(total * args.ratios[1]) if total >= 3 else 0
    n_train = max(n_train, 1)

    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }

    empty_images = 0
    for split, items in splits.items():
        for image in items:
            target_image = out_root / "images" / split / image.name
            target_label = out_root / "labels" / split / f"{image.stem}.txt"
            if args.move:
                shutil.move(str(image), target_image)
            else:
                shutil.copy2(image, target_image)
            rows = labels_for(image)
            if not rows:
                empty_images += 1
            text = "".join(
                f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cls, cx, cy, w, h in rows
            )
            target_label.write_text(text, encoding="utf-8")

    import yaml

    config = repo_path("configs") / f"dataset.{args.slug}.yaml"
    config.write_text(
        "# 自建数据集，由 scripts/import_custom_dataset.py 生成\n"
        + yaml.safe_dump(
            {
                "path": str(out_root),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {index: name for index, name in enumerate(classes)},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    print(f"输出: {out_root}")
    for split, items in splits.items():
        print(f"  {split:<6}{len(items):>4} 张")
    if empty_images:
        print(f"其中 {empty_images} 张没有标注（作为负样本，这是好事）")
    print(f"配置: {config}")
    print(f"下一步: python -m src.prepare stats --config configs/dataset.{args.slug}.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
