"""把 YOLO 格式的标签读成 torchvision 检测模型要的 (image, target)。"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def polygon_to_bbox(values: list[float]) -> tuple[float, float, float, float]:
    """标签可能是 4 个数（框）也可能是 2n 个数（多边形），统一转成 cxcywh。"""
    if len(values) == 4:
        cx, cy, w, h = values
        return cx, cy, w, h
    xs, ys = values[0::2], values[1::2]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)


def read_yolo_labels(label_path: Path, width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (boxes[N,4] xyxy 像素坐标, labels[N])，label 已经 +1（0 留给背景）。"""
    boxes: list[list[float]] = []
    labels: list[int] = []
    if not label_path.exists():
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = polygon_to_bbox([float(v) for v in parts[1:]])
        if w <= 1e-6 or h <= 1e-6:
            continue
        x0 = max(0.0, (cx - w / 2) * width)
        y0 = max(0.0, (cy - h / 2) * height)
        x1 = min(float(width), (cx + w / 2) * width)
        y1 = min(float(height), (cy + h / 2) * height)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        boxes.append([x0, y0, x1, y1])
        labels.append(cls + 1)

    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)
    return (
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.int64),
    )


class YoloDetectionDataset(Dataset):
    def __init__(
        self, root: Path, split: str, min_side: int | None = None, augment: bool = False
    ) -> None:
        self.image_dir = root / "images" / split
        self.label_dir = root / "labels" / split
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"找不到图片目录: {self.image_dir}")
        self.images = sorted(p for p in self.image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        self.min_side = min_side
        # 增强只给训练集开；验证集必须保持原样，否则指标不可比
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        path = self.images[index]
        image = Image.open(path).convert("RGB")
        if self.min_side and min(image.size) < self.min_side:
            scale = self.min_side / min(image.size)
            image = image.resize((int(image.width * scale), int(image.height * scale)))

        # 水平翻转 + 轻度色彩扰动。缺陷大多没有方向性，翻转是安全的；
        # 不做旋转/裁剪，避免把细长划痕切碎后引入错误标签。
        flipped = False
        if self.augment:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
                flipped = True
            if random.random() < 0.4:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
            if random.random() < 0.4:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))

        width, height = image.size
        boxes, labels = read_yolo_labels(self.label_dir / f"{path.stem}.txt", width, height)
        if flipped and boxes.numel():
            x0, x1 = boxes[:, 0].clone(), boxes[:, 2].clone()
            boxes[:, 0] = width - x1
            boxes[:, 2] = width - x0

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
        }
        image_tensor = torch.from_numpy(
            np.asarray(image, dtype="float32") / 255.0
        ).permute(2, 0, 1).contiguous()
        return image_tensor, target


def collate_fn(batch):
    images, targets = zip(*batch, strict=False)
    return list(images), list(targets)
