"""数据体检与切分。

训练前先跑一遍 stats / dups。经验上"训不动"的原因里，数据问题比模型问题多得多：
标签越界、类别写错、验证集和训练集有重复图（泄漏）、负样本没建空标签。
这些问题在训练曲线上只表现为"精度上不去"，很难反查。

用法：
    python -m src.prepare stats
    python -m src.prepare dups
    python -m src.prepare split --src data/raw --out data/dataset
"""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.config import load_dataset, repo_path

SPLITS = ("train", "val", "test")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(slots=True)
class SplitStats:
    split: str
    images: int = 0
    labels: int = 0
    missing_labels: int = 0
    empty_labels: int = 0
    boxes: int = 0
    per_class: Counter = field(default_factory=Counter)


def iter_images(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        return []
    return sorted(
        p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def label_for(image: Path, images_root: Path, labels_root: Path) -> Path:
    """images/train/a.jpg -> labels/train/a.txt"""
    return labels_root / image.relative_to(images_root).with_suffix(".txt")


def parse_label_file(path: Path, num_classes: int) -> list[tuple[int, list[float]]]:
    """解析 YOLO 标签文件，返回 [(class_id, [坐标...]), ...]。

    检测是 5 个字段，分割是 1 + 2n 个字段（n 个多边形点），都是奇数个，
    所以"字段数为偶数"本身就是一个可靠的格式错误信号。
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"标签不是 UTF-8 文本（{exc}）") from exc

    boxes: list[tuple[int, list[float]]] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"第 {line_no} 行只有 {len(parts)} 个字段，至少需要 5 个")
        if len(parts) % 2 == 0:
            raise ValueError(f"第 {line_no} 行字段数为 {len(parts)}，应为奇数（1 + 2n）")
        try:
            cls = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"第 {line_no} 行含非数字内容: {line[:60]}") from exc
        if not 0 <= cls < num_classes:
            raise ValueError(f"第 {line_no} 行类别 id={cls} 超出范围 0..{num_classes - 1}")
        if any(v < 0.0 or v > 1.0 for v in coords):
            raise ValueError(f"第 {line_no} 行坐标未归一化到 0~1: {coords}")
        if len(coords) == 4 and (coords[2] <= 0 or coords[3] <= 0):
            raise ValueError(f"第 {line_no} 行框宽或高为 0")
        boxes.append((cls, coords))
    return boxes


def validate_dataset(root: Path, names: dict[int, str]) -> tuple[dict[str, SplitStats], list[str]]:
    """遍历数据集，返回各 split 的统计和错误列表（不抛异常，尽量一次报全）。"""
    images_root = root / "images"
    labels_root = root / "labels"
    stats: dict[str, SplitStats] = {}
    errors: list[str] = []

    for split in SPLITS:
        image_dir = images_root / split
        if not image_dir.is_dir():
            continue
        current = SplitStats(split=split)
        for image in iter_images(image_dir):
            current.images += 1
            label = label_for(image, images_root, labels_root)
            if not label.exists():
                current.missing_labels += 1
                errors.append(f"[{split}] 缺标签文件: {label.relative_to(root)}")
                continue
            current.labels += 1
            try:
                boxes = parse_label_file(label, num_classes=len(names))
            except ValueError as exc:
                errors.append(f"[{split}] {label.relative_to(root)}: {exc}")
                continue
            if not boxes:
                current.empty_labels += 1  # 负样本：合法，但要心里有数
            current.boxes += len(boxes)
            for cls, _ in boxes:
                current.per_class[names[cls]] += 1
        stats[split] = current

    if not stats:
        errors.append(f"在 {images_root} 下没找到 train/val/test 任何目录")
    return stats, errors


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def find_duplicates(root: Path) -> dict[str, list[Path]]:
    """按内容哈希找重复图。跨 split 的重复 = 数据泄漏，验证指标会虚高。"""
    buckets: dict[str, list[Path]] = defaultdict(list)
    for image in iter_images(root / "images"):
        buckets[file_md5(image)].append(image)
    return {key: sorted(paths) for key, paths in buckets.items() if len(paths) > 1}


def write_dataset_yaml(root: Path, names: dict[int, str], path_value: str | None = None) -> Path:
    import yaml

    payload = {
        "path": path_value or str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {int(k): v for k, v in names.items()},
    }
    target = root / "dataset.yaml"
    target.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return target


def split_dataset(
    src_root: Path,
    out_root: Path,
    names: dict[int, str],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    move: bool = False,
) -> dict[str, int]:
    """把平铺的 images/ + labels/ 随机切成 train/val/test。"""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios 之和必须为 1，当前 {ratios}")
    images = iter_images(src_root / "images")
    if not images:
        raise ValueError(f"{src_root / 'images'} 下没有图片")

    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    if total < 3:
        # 图太少时切验证集没有意义，全部给 train
        n_train, n_val = total, 0
    else:
        n_train = max(int(total * ratios[0]), 1)
        n_val = max(int(total * ratios[1]), 1)
        if n_train + n_val > total:
            n_train = total - n_val
    n_test = total - n_train - n_val
    chunks = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val : n_train + n_val + n_test],
    }

    counts: dict[str, int] = {}
    for split, items in chunks.items():
        if not items:
            continue
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image in items:
            target_image = out_root / "images" / split / image.name
            target_label = out_root / "labels" / split / f"{image.stem}.txt"
            if target_image.exists():
                raise FileExistsError(f"目标已存在，拒绝覆盖: {target_image}")
            source_label = src_root / "labels" / f"{image.stem}.txt"
            if move:
                shutil.move(str(image), target_image)
                if source_label.exists():
                    shutil.move(str(source_label), target_label)
            else:
                shutil.copy2(image, target_image)
                target_label.write_text(
                    source_label.read_text(encoding="utf-8") if source_label.exists() else "",
                    encoding="utf-8",
                )
        counts[split] = len(items)

    write_dataset_yaml(out_root, names)
    return counts


def format_report(stats: dict[str, SplitStats], names: dict[int, str]) -> str:
    lines: list[str] = []
    lines.append(f"{'split':<6}{'images':>8}{'labels':>8}{'缺标签':>8}{'负样本':>8}{'框数':>8}")
    lines.append("-" * 46)
    for split, st in stats.items():
        lines.append(
            f"{split:<6}{st.images:>8}{st.labels:>8}{st.missing_labels:>8}"
            f"{st.empty_labels:>8}{st.boxes:>8}"
        )
    lines.append("")
    merged: Counter = Counter()
    for st in stats.values():
        merged.update(st.per_class)
    for index in sorted(names):
        lines.append(f"  {index}  {names[index]:<12}{merged.get(names[index], 0):>7}")
    return "\n".join(lines)


def _resolve_root(cfg: dict) -> Path:
    return repo_path(cfg["path"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.prepare", description="数据体检与切分")
    # 子命令共享 --config，这样 `prepare stats --config x.yaml` 也成立
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/dataset.yaml", help="数据集配置路径")
    sub = parser.add_subparsers(dest="command", required=True)

    stats_cmd = sub.add_parser("stats", parents=[common], help="检查标签格式并统计类别分布")
    stats_cmd.add_argument("--max-errors", type=int, default=20)

    sub.add_parser("dups", parents=[common], help="按内容哈希查重复图与跨 split 泄漏")

    split_cmd = sub.add_parser("split", parents=[common], help="把平铺数据切成 train/val/test")
    split_cmd.add_argument("--src", required=True, help="含 images/ 与 labels/ 的源目录")
    split_cmd.add_argument("--out", required=True, help="输出数据集根目录")
    split_cmd.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    split_cmd.add_argument("--seed", type=int, default=42)
    split_cmd.add_argument("--move", action="store_true", help="移动而非复制（省磁盘）")

    args = parser.parse_args(argv)
    cfg = load_dataset(args.config)
    names: dict[int, str] = cfg["names"]

    if args.command == "stats":
        root = _resolve_root(cfg)
        stats, errors = validate_dataset(root, names)
        print(f"数据集: {root}")
        print(format_report(stats, names))
        if errors:
            print(f"\n发现 {len(errors)} 个问题（只显示前 {args.max_errors} 条）:")
            for message in errors[: args.max_errors]:
                print(f"  - {message}")
            return 1
        print("\n标签格式检查通过。")
        return 0

    if args.command == "dups":
        root = _resolve_root(cfg)
        duplicates = find_duplicates(root)
        if not duplicates:
            print("没有重复图片。")
            return 0
        leaked = 0
        for paths in duplicates.values():
            splits = {p.relative_to(root).parts[1] for p in paths if "images" in p.parts}
            flag = ""
            if len(splits) > 1:
                leaked += 1
                flag = "  <-- 跨 split 泄漏"
            print(f"{len(paths)} 张重复{flag}")
            for path in paths:
                print(f"    {path.relative_to(root)}")
        print(f"\n共 {len(duplicates)} 组重复，其中 {leaked} 组跨 split。")
        return 1 if leaked else 0

    out_root = repo_path(args.out)
    counts = split_dataset(
        src_root=repo_path(args.src),
        out_root=out_root,
        names=names,
        ratios=tuple(args.ratios),  # type: ignore[arg-type]
        seed=args.seed,
        move=args.move,
    )
    print(f"切分完成 -> {out_root}")
    for split, count in counts.items():
        print(f"  {split:<6}{count:>6}")
    print(f"已写入 {out_root / 'dataset.yaml'}（记得把 configs/dataset.yaml 的 path 指过去）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
