"""判级阈值校准：在不改变"参考标准"的前提下，调系统侧的判定阈值。

关键设计：
    参考等级 = 按**开题原始阈值**从标注算出来的等级 —— 全程固定，不参与优化；
    系统等级 = 按**候选阈值**从模型检测结果算出来的等级；
    目标     = 让系统等级尽可能贴近参考等级。

如果连参考等级一起调，那就是"让标准去迁就模型"，校准就变成了自欺。
所以这里只动系统侧，且**在 val 上拟合、在 test 上验证**。

    python -m src.calibrate_grading --weights runs/yolov8s/weights/best.pt
    python -m src.calibrate_grading --weights runs/baseline/weights/best.pt --tag yolo11n
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys

import numpy as np

from src.config import load_dataset, repo_path
from src.detectors import load_detector
from src.evaluate_grading import label_to_detections
from src.grade import grade, load_grading_config


def collect(weights: str, dataset: str, split: str, conf: float):
    """跑一遍模型，把每张图的 (参考等级, 检测结果, 图像尺寸) 缓存下来，供网格搜索复用。"""
    cfg = load_dataset(dataset)
    root = repo_path(cfg["path"])
    names = cfg["names"]
    base = load_grading_config()
    detector = load_detector(weights)
    use_mask = getattr(getattr(detector, "model", None), "task", "") == "segment"

    from PIL import Image

    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise SystemExit(f"{image_dir} 下没有图片")

    items = []
    for path in images:
        image = np.asarray(Image.open(path).convert("RGB"))
        height, width = image.shape[:2]
        gt = label_to_detections(label_dir / f"{path.stem}.txt", width, height, names, with_mask=use_mask)
        # 参考等级用**原始配置**，一旦算好就不再随候选参数变化
        reference = grade(gt, (width, height), base).level
        result = detector.predict(image, conf=conf)
        items.append((reference, result.detections, result.image_size))
    return items, base


def score_config(items, config) -> dict:
    confusion = np.zeros((5, 5), dtype=int)
    for reference, detections, size in items:
        predicted = grade(detections, size, config).level
        confusion[reference, predicted] += 1
    total = int(confusion.sum())
    exact = float(np.trace(confusion)) / total
    within = float(sum(confusion[i][j] for i in range(5) for j in range(5) if abs(i - j) <= 1)) / total
    iv_truth = int(confusion[4].sum())
    iv_recall = float(confusion[4][4]) / iv_truth if iv_truth else 1.0
    low = int(confusion[0].sum() + confusion[1].sum())
    overkill = float(confusion[0][3] + confusion[0][4] + confusion[1][3] + confusion[1][4]) / low if low else 0.0
    return {
        "exact": exact,
        "within_one": within,
        "iv_recall": iv_recall,
        "overkill": overkill,
        "confusion": confusion.tolist(),
        "total": total,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.calibrate_grading", description="判级阈值校准")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", default="configs/dataset.msdd.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--fit-split", default="val", choices=["train", "val"])
    parser.add_argument("--eval-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--min-iv-recall", type=float, default=0.90,
                        help="IV 级召回的下限约束，避免用漏判换一致率")
    args = parser.parse_args(argv)

    print(f"拟合集 {args.fit_split}，验证集 {args.eval_split}")
    fit_items, base = collect(args.weights, args.dataset, args.fit_split, args.conf)
    eval_items, _ = collect(args.weights, args.dataset, args.eval_split, args.conf)

    default_fit = score_config(fit_items, base)
    print(f"默认阈值（{args.fit_split}）: 一致率 {default_fit['exact']*100:.1f}%  "
          f"误杀 {default_fit['overkill']*100:.1f}%  IV召回 {default_fit['iv_recall']*100:.1f}%")

    area_grid = [0.003, 0.005, 0.008]
    area3_grid = [0.015, 0.020, 0.030]
    area4_grid = [0.040, 0.050, 0.070]
    count2_grid = [2, 3]
    count3_grid = [5, 6]
    count4_grid = [9, 11]

    best = None
    tried = 0
    for a2, a3, a4, c2, c3, c4 in itertools.product(
        area_grid, area3_grid, area4_grid, count2_grid, count3_grid, count4_grid
    ):
        if not (a2 < a3 < a4):
            continue
        if not (c2 < c3 < c4):
            continue
        candidate = copy.deepcopy(base)
        candidate["criteria"]["area_ratio"] = {"level2": a2, "level3": a3, "level4": a4}
        candidate["criteria"]["count"] = {"level2": c2, "level3": c3, "level4": c4}
        try:
            metrics = score_config(fit_items, candidate)
        except Exception:  # noqa: BLE001 - 个别参数组合可能让规则退化
            continue
        tried += 1
        # 先满足 IV 召回约束，再比一致率；同分时用 ±1 一致率做次级排序
        feasible = metrics["iv_recall"] >= args.min_iv_recall
        key = (feasible, round(metrics["exact"], 4), round(metrics["within_one"], 4))
        if best is None or key > best[0]:
            best = (key, metrics, candidate)

    if best is None:
        raise SystemExit("没有找到可行参数组合")
    _, fit_metrics, best_cfg = best
    print(f"搜索 {tried} 组；最优（{args.fit_split}）: 一致率 {fit_metrics['exact']*100:.1f}%  "
          f"误杀 {fit_metrics['overkill']*100:.1f}%  IV召回 {fit_metrics['iv_recall']*100:.1f}%")

    eval_default = score_config(eval_items, base)
    eval_best = score_config(eval_items, best_cfg)
    print(f"\n{args.eval_split} 集对比：")
    print(f"  校准前: 一致率 {eval_default['exact']*100:.1f}%  误杀 {eval_default['overkill']*100:.1f}%  "
          f"IV召回 {eval_default['iv_recall']*100:.1f}%")
    print(f"  校准后: 一致率 {eval_best['exact']*100:.1f}%  误杀 {eval_best['overkill']*100:.1f}%  "
          f"IV召回 {eval_best['iv_recall']*100:.1f}%")

    report = {
        "tag": args.tag,
        "weights": args.weights,
        "fit_split": args.fit_split,
        "eval_split": args.eval_split,
        "conf": args.conf,
        "grid_size": tried,
        "constraint_min_iv_recall": args.min_iv_recall,
        "fit_default": default_fit,
        "fit_best": fit_metrics,
        "eval_default": eval_default,
        "eval_best": eval_best,
        "best_criteria": {
            "area_ratio": best_cfg["criteria"]["area_ratio"],
            "count": best_cfg["criteria"]["count"],
        },
    }
    out = repo_path("results") / f"calibration_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    import yaml

    calibrated = repo_path("configs") / f"grading.calibrated.{args.tag}.yaml"
    calibrated.write_text(
        f"# 由 src/calibrate_grading.py 在 {args.fit_split} 上拟合、{args.eval_split} 上验证\n"
        + yaml.safe_dump(best_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"\n报告: {out}\n配置: {calibrated}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
