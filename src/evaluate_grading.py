"""分级实验：检测精度如何传递到判级准确率。

思路（也是答辩老师说的"很少有人做的量化分级"）：

1. 用**标注框**按 configs/grading.yaml 的规则算出"参考等级"——相当于完美检测下的判级；
2. 用**模型预测框**算"系统等级"；
3. 比较两者：完全一致率、±1 级内一致率、判废（IV 级）召回率、混淆矩阵。

这样一来，"mAP 高 0.05 到底值不值得"就有了下游口径的回答。

    python -m src.evaluate_grading --weights runs/baseline/weights/best.pt --split test
    python -m src.evaluate_grading --weights runs/cnn-cbam/best.pt --split test --tag cnn-grading
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.config import load_dataset, repo_path
from src.detectors import Detection, load_detector
from src.grade import grade, load_grading_config

LEVELS = [0, 1, 2, 3, 4]
# 矩阵表头用短标签，0 表示"未检出缺陷"
LEVEL_LABELS = {0: "合格", 1: "I", 2: "II", 3: "III", 4: "IV"}


def label_to_detections(
    label_path: Path, width: int, height: int, names: dict[int, str], with_mask: bool = True
) -> list[Detection]:
    """把标注转成 Detection。with_mask=True 时多边形会栅格化成掩码。

    **口径必须和被测模型一致**：框模型（YOLO 检测）就用框算面积，
    掩码模型（YOLO-seg）就用掩码算。混着比会引入系统性偏差。
    """
    detections: list[Detection] = []
    if not label_path.exists():
        return detections

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        coords = [float(v) for v in parts[1:]]
        mask = None
        if with_mask and len(coords) > 4:  # 多边形：还原成二值掩码
            from PIL import Image, ImageDraw

            canvas = Image.new("L", (width, height), 0)
            points = [
                (coords[i] * width, coords[i + 1] * height) for i in range(0, len(coords) - 1, 2)
            ]
            ImageDraw.Draw(canvas).polygon(points, fill=1)
            mask = np.asarray(canvas, dtype=bool)

        xs, ys = coords[0::2], coords[1::2]
        box = (
            max(0.0, min(xs) * width),
            max(0.0, min(ys) * height),
            min(float(width), max(xs) * width),
            min(float(height), max(ys) * height),
        )
        detections.append(
            Detection(
                cls_id=cls,
                cls_name=names.get(cls, str(cls)),
                confidence=1.0,
                box=box,
                mask=mask,
            )
        )
    return detections


def reference_grade(
    image_path: Path, label_path: Path, image_size, grading_cfg, with_mask: bool
) -> int:
    """用标注算"参考等级"，判据口径与模型输出一致。"""
    width, height = image_size
    names: dict[int, str] = grading_cfg["names_lookup"]
    detections = label_to_detections(label_path, width, height, names, with_mask=with_mask)
    return grade(detections, image_size, grading_cfg["config"]).level


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.evaluate_grading", description="分级一致性实验")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", default="configs/dataset.msdd.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", default="results")
    args = parser.parse_args(argv)

    dataset_cfg = load_dataset(args.dataset)
    root = repo_path(dataset_cfg["path"])
    grading_cfg = load_grading_config()
    names = dataset_cfg["names"]
    payload = {"config": grading_cfg, "names_lookup": names}

    # 先探一下模型输出有没有掩码，决定参考等级的算法口径（必须一致）
    from PIL import Image

    image_dir = root / "images" / args.split
    label_dir = root / "labels" / args.split
    images = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not images:
        raise SystemExit(f"{image_dir} 下没有图片")

    detector = load_detector(args.weights)
    probe_image = np.asarray(Image.open(images[0]).convert("RGB"))
    probe = detector.predict(probe_image, conf=args.conf)
    with_mask = any(d.mask is not None for d in probe.detections)
    print(f"口径: {'掩码面积' if with_mask else '框面积'}（与模型输出一致）")

    confusion = np.zeros((5, 5), dtype=int)  # 行=参考等级, 列=系统等级
    rows = []
    for path in images:
        image = np.asarray(Image.open(path).convert("RGB"))
        height, width = image.shape[:2]
        truth = reference_grade(
            path, label_dir / f"{path.stem}.txt", (width, height), payload, with_mask
        )
        result = detector.predict(image, conf=args.conf)
        predicted = grade(result.detections, result.image_size).level
        confusion[truth, predicted] += 1
        rows.append({"image": path.name, "truth": truth, "predicted": predicted})

    total = int(confusion.sum())
    exact = int(np.trace(confusion))
    within_one = int(sum(confusion[i][j] for i in range(5) for j in range(5) if abs(i - j) <= 1))

    # 判废（IV 级）召回：参考为 IV 的图里，系统也判 IV 的比例——漏判代价最大
    iv_truth = int(confusion[4].sum())
    iv_hit = int(confusion[4][4])
    iv_recall = iv_hit / iv_truth if iv_truth else None

    # 把参考为"合格/轻微"的判成 III 级以上，属于"误杀"，产线上同样有成本
    overkill = int(confusion[0][3] + confusion[0][4] + confusion[1][3] + confusion[1][4])
    low_truth = int(confusion[0].sum() + confusion[1].sum())

    summary = {
        "tag": args.tag,
        "weights": args.weights,
        "split": args.split,
        "conf": args.conf,
        "images": total,
        "exact_accuracy": exact / total if total else 0.0,
        "within_one_accuracy": within_one / total if total else 0.0,
        "iv_recall": iv_recall,
        "overkill_rate": overkill / low_truth if low_truth else None,
        "confusion": confusion.tolist(),
        "rows": rows,
    }

    out_dir = repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"grading_{args.tag}.json"
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 终端里直接打印表格，方便贴进论文
    print(f"分级一致性（{args.split}，{total} 张，conf={args.conf}）")
    print("参考\\系统" + "".join(f"{LEVEL_LABELS[i]:>7}" for i in LEVELS))
    for i in LEVELS:
        cells = "".join(f"{confusion[i][j]:>8d}" for j in LEVELS)
        print(f"{LEVEL_LABELS[i]:<8}{cells}")
    print()
    print(f"完全一致率      {summary['exact_accuracy'] * 100:.1f}%")
    print(f"±1 级内一致率   {summary['within_one_accuracy'] * 100:.1f}%")
    if iv_recall is not None:
        print(f"IV 级召回率     {iv_recall * 100:.1f}%  ({iv_hit}/{iv_truth})")
    if summary["overkill_rate"] is not None:
        print(f"低等级误杀率    {summary['overkill_rate'] * 100:.1f}%")
    print(f"已写入 {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
