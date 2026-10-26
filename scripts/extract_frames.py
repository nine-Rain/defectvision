"""从视频里抽帧，做成待标注的原始图片集。

用于"自建数据集"的素材采集：把手机拍的、或者下载的工业零件缺陷视频，
按固定间隔抽帧、去重、过滤模糊帧，输出一组可以直接丢进 Label Studio 的图片。

    python scripts/extract_frames.py --video D:/videos/parts.mp4 --out data/custom_raw
    python scripts/extract_frames.py --video-dir D:/videos --out data/custom_raw --every 0.5

抽帧的同时会写一个 provenance.csv，记录每张图来自哪个视频、第几秒——
论文里交代数据来源时要用，答辩被问"你这数据哪来的"也能当场答上。

注意：视频素材的版权由采集者负责。用于学位论文时建议优先使用自己拍摄的视频，
或已获授权的素材，并在 provenance.csv 里注明来源与授权情况。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import repo_path  # noqa: E402


def dhash(image: np.ndarray, size: int = 8) -> int:
    """差值哈希：缩到 9x8 灰度，比较相邻像素大小，得到 64 位指纹。

    比均值哈希（aHash）更能区分"同一个零件的不同角度"——实测同一批次的两张图
    在 aHash 下距离为 0（被误判成重复），在 dHash 下是 14，能正确保留。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def is_blurry(image: np.ndarray, threshold: float) -> bool:
    """拉普拉斯方差越小越模糊。手机拍摄的虚焦帧对标注没价值，直接丢掉。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < threshold


def iter_videos(video: Path | None, video_dir: Path | None) -> list[Path]:
    if video:
        return [video]
    suffixes = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}
    assert video_dir is not None
    return sorted(p for p in video_dir.rglob("*") if p.suffix.lower() in suffixes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从视频抽帧做自建数据集")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="单个视频文件")
    group.add_argument("--video-dir", help="视频目录（递归）")
    parser.add_argument("--out", default="data/custom_raw")
    parser.add_argument("--every", type=float, default=0.5, help="每隔多少秒抽一帧")
    parser.add_argument("--max-frames", type=int, default=400, help="最多输出多少张")
    parser.add_argument("--min-side", type=int, default=480, help="短边小于这个值就跳过")
    parser.add_argument("--blur-threshold", type=float, default=60.0, help="低于此值视为模糊帧")
    parser.add_argument("--dedupe-distance", type=int, default=5, help="哈希汉明距离小于此值判为重复（dHash 64 位）")
    args = parser.parse_args(argv)

    videos = iter_videos(
        Path(args.video) if args.video else None,
        Path(args.video_dir) if args.video_dir else None,
    )
    if not videos:
        raise SystemExit("没有找到视频文件")

    out_dir = repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    hashes: list[int] = []
    stats = {"read": 0, "kept": 0, "blurry": 0, "small": 0, "duplicate": 0}

    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"跳过（打不开）: {video_path}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(fps * args.every)))
        frame_index = 0
        print(f"处理 {video_path.name}  fps={fps:.1f}  每 {step} 帧取一张")

        while len(rows) < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step == 0:
                stats["read"] += 1
                height, width = frame.shape[:2]
                if min(height, width) < args.min_side:
                    stats["small"] += 1
                elif is_blurry(frame, args.blur_threshold):
                    stats["blurry"] += 1
                else:
                    fingerprint = dhash(frame)
                    if any(hamming(fingerprint, old) <= args.dedupe_distance for old in hashes):
                        stats["duplicate"] += 1
                    else:
                        hashes.append(fingerprint)
                        name = f"{video_path.stem}_{frame_index:06d}.jpg"
                        cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        rows.append(
                            {
                                "file": name,
                                "source": str(video_path),
                                "frame_index": frame_index,
                                "second": round(frame_index / fps, 2),
                                "width": width,
                                "height": height,
                                "license": "待填写（自拍则填“本人拍摄”）",
                                "note": "",
                            }
                        )
                        stats["kept"] += 1
            frame_index += 1
        cap.release()

    csv_path = out_dir / "provenance.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file", "source", "frame_index", "second", "width", "height", "license", "note"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\n候选 {stats['read']} 帧 -> 保留 {stats['kept']} 张"
        f"（模糊 {stats['blurry']}、过小 {stats['small']}、重复 {stats['duplicate']}）"
    )
    print(f"输出目录: {out_dir}")
    print(f"来源记录: {csv_path}")
    print("下一步：用 Label Studio 标注后，跑 scripts/import_custom_dataset.py 接入数据集")
    return 0


if __name__ == "__main__":
    sys.exit(main())
