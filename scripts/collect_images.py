"""把"从浏览器/图库里下载的图片"整理成待标注的原始图片集。

两种用法：

    # 1. 已经下载到文件夹（浏览器右键另存、素材站批量下载都行）
    python scripts/collect_images.py --src D:/downloads/parts --out data/custom_raw

    # 2. 手里有一串图片直链
    python scripts/collect_images.py --urls urls.txt --out data/custom_raw

会做三件事：校验能正常打开、按内容去重（均值哈希）、统一命名并写 provenance.csv。
provenance.csv 里的 license / source_url 两列**需要你补全**——论文里要交代数据来源，
答辩也一定会被问。建议只用来路清晰、允许使用的素材。
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_frames import dhash, hamming  # noqa: E402

from src.config import repo_path  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
UA = {"User-Agent": "Mozilla/5.0 (research dataset collection)"}


def download(url: str, target: Path, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read()
            if len(data) < 4096:
                return False
            target.write_bytes(data)
            return True
        except Exception:  # noqa: BLE001 - 单张失败不该中断整批
            time.sleep(1.5 * (attempt + 1))
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="整理浏览器下载的图片为待标注素材")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--src", help="已下载图片的文件夹（递归）")
    group.add_argument("--urls", help="每行一个图片直链的文本文件")
    parser.add_argument("--out", default="data/custom_raw")
    parser.add_argument("--max-side", type=int, default=1600, help="超过则等比缩小，控制体积")
    parser.add_argument("--dedupe-distance", type=int, default=5)
    args = parser.parse_args(argv)

    out_dir = repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = out_dir / "_incoming"
    staging.mkdir(exist_ok=True)

    candidates: list[tuple[Path, str]] = []  # (本地路径, 来源说明)

    if args.src:
        src = Path(args.src)
        if not src.exists():
            raise SystemExit(f"目录不存在: {src}")
        for path in sorted(src.rglob("*")):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                candidates.append((path, str(path)))
    else:
        url_file = Path(args.urls)
        if not url_file.exists():
            raise SystemExit(f"找不到 URL 列表: {url_file}")
        urls = [line.strip() for line in url_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        for index, url in enumerate(urls):
            suffix = Path(url.split("?")[0]).suffix.lower()
            suffix = suffix if suffix in IMAGE_SUFFIXES else ".jpg"
            target = staging / f"url_{index:04d}{suffix}"
            if download(url, target):
                candidates.append((target, url))
            else:
                print(f"下载失败: {url}")

    rows: list[dict] = []
    hashes: list[int] = []
    stats = {"total": len(candidates), "bad": 0, "duplicate": 0, "kept": 0}

    for path, source in candidates:
        try:
            with Image.open(path) as probe:
                probe.verify()
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError("cv2 读不出来")
        except Exception as exc:  # noqa: BLE001 - 坏文件直接跳过
            print(f"跳过损坏文件 {path.name}: {exc}")
            stats["bad"] += 1
            continue

        fingerprint = dhash(image)
        if any(hamming(fingerprint, old) <= args.dedupe_distance for old in hashes):
            stats["duplicate"] += 1
            continue
        hashes.append(fingerprint)

        height, width = image.shape[:2]
        if max(height, width) > args.max_side:
            scale = args.max_side / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
            height, width = image.shape[:2]

        name = f"collected_{stats['kept']:04d}.jpg"
        cv2.imwrite(str(out_dir / name), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append(
            {
                "file": name,
                "source": source,
                "width": width,
                "height": height,
                "license": "待填写",
                "note": "",
            }
        )
        stats["kept"] += 1

    shutil.rmtree(staging, ignore_errors=True)

    csv_path = out_dir / "provenance.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "source", "width", "height", "license", "note"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n候选 {stats['total']} 张 -> 保留 {stats['kept']} 张"
          f"（损坏 {stats['bad']}、重复 {stats['duplicate']}）")
    print(f"输出目录: {out_dir}")
    print(f"来源记录: {csv_path}  ← 请补全 license 列")
    print("下一步：Label Studio 标注 -> scripts/import_custom_dataset.py 接入")
    return 0


if __name__ == "__main__":
    sys.exit(main())
