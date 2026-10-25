"""评估与实验对比。

每次评估落一个 results/<tag>.json，再自动汇总成 results/metrics.md ——
这张表就是你 README 里最值钱的部分，也是面试时唯一能被追问的"证据"。

    python -m src.evaluate --weights runs/baseline/weights/best.pt --tag baseline
    python -m src.evaluate --weights runs/finetune/weights/best.pt --tag finetune
    python -m src.evaluate --compare          # 只重新生成对比表
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import repo_path

RESULT_KEYS = {
    "mAP50-95": ("mAP50-95",),
    "mAP50": ("mAP50(B)", "mAP50(M)", "mAP50"),
    "precision": ("precision(B)", "precision(M)", "precision"),
    "recall": ("recall(B)", "recall(M)", "recall"),
}


def _pick(results_dict: dict[str, Any], needles: tuple[str, ...]) -> float | None:
    """在 results_dict 里按子串匹配取值，兼容检测 (B) 与分割 (M) 的键名。"""
    for key, value in results_dict.items():
        if all(needle in key for needle in needles):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def evaluate(weights: Path, data: str, split: str, conf: float, iou: float) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(data=str(repo_path(data)), split=split, conf=conf, iou=iou, verbose=False)
    raw: dict[str, Any] = dict(getattr(metrics, "results_dict", {}) or {})

    params = None
    model_obj = getattr(model, "model", None)
    if model_obj is not None:
        try:
            params = int(sum(p.numel() for p in model_obj.parameters()))
        except Exception:  # noqa: BLE001 - 导出模型可能没有 parameters()
            params = None

    return {
        "weights": str(weights),
        "size_mb": round(weights.stat().st_size / (1024 * 1024), 2) if weights.exists() else None,
        "params": params,
        "split": split,
        "raw": raw,
        "mAP50-95": _pick(raw, RESULT_KEYS["mAP50-95"]),
        "mAP50": _pick(raw, RESULT_KEYS["mAP50"]),
        "precision": _pick(raw, RESULT_KEYS["precision"]),
        "recall": _pick(raw, RESULT_KEYS["recall"]),
        "inference_ms": (getattr(metrics, "speed", {}) or {}).get("inference"),
    }


def save_result(result: dict[str, Any], tag: str, note: str, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": tag,
        "note": note,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        **result,
    }
    target = results_dir / f"{tag}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def write_table(results_dir: Path) -> str:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    if not rows:
        return "还没有评估结果，先跑一次 src.evaluate。"

    rows.sort(key=lambda r: (r.get("mAP50-95") or -1), reverse=True)
    lines = [
        "| 方案 | mAP50-95 | mAP50 | Precision | Recall | 体积 (MB) | 推理 (ms) | 备注 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {tag} | {map95} | {map50} | {p} | {r} | {size} | {ms} | {note} |".format(
                tag=row.get("tag", "?"),
                map95=_fmt(row.get("mAP50-95")),
                map50=_fmt(row.get("mAP50")),
                p=_fmt(row.get("precision")),
                r=_fmt(row.get("recall")),
                size=_fmt(row.get("size_mb"), 2),
                ms=_fmt(row.get("inference_ms"), 2),
                note=row.get("note", ""),
            )
        )
    table = "\n".join(lines)
    (results_dir / "metrics.md").write_text(table + "\n", encoding="utf-8")
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.evaluate", description="评估并生成对比表")
    parser.add_argument("--weights")
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--tag", help="实验名，默认取权重所在目录名")
    parser.add_argument("--note", default="", help="写进对比表的备注")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--compare", action="store_true", help="只根据已有 json 重建对比表")
    args = parser.parse_args(argv)

    results_dir = repo_path(args.results_dir)

    if not args.compare:
        if not args.weights:
            parser.error("需要 --weights，或者用 --compare 只重建表格")
        weights = repo_path(args.weights)
        if not weights.exists():
            raise SystemExit(f"找不到权重文件: {weights}")
        tag = args.tag or weights.parent.parent.name
        result = evaluate(weights, args.data, args.split, args.conf, args.iou)
        path = save_result(result, tag, args.note, results_dir)
        print(f"结果已写入 {path}")
        print(f"  mAP50-95 = {_fmt(result['mAP50-95'])}   mAP50 = {_fmt(result['mAP50'])}")

    table = write_table(results_dir)
    print()
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
