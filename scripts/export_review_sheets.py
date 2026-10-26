"""导出"判级争议样本"复核图，用于人工核对是模型误检还是标签漏标。

筛选规则：模型判出的等级比"参考等级"高 2 级及以上（例如标注合格、模型判 IV）。
这类样本是误杀率的主要来源，也是最值得人工看的一批。

    python scripts/export_review_sheets.py `
        --grading results/grading_rtdetr-ab-binary.json results/grading_cnn-ab-binary.json `
        --out docs/review --limit 50

输出：
    docs/review/sheet_01.png ...   每张 4 个样本，红框=模型预测，绿框=标注
    docs/review/index.csv          待填表格：case_id / 文件 / 等级 / 结论(留空给你填)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_dataset, repo_path  # noqa: E402
from src.detectors import load_detector  # noqa: E402
from src.evaluate_grading import label_to_detections  # noqa: E402

LEVEL_LABEL = {0: "合格", 1: "I", 2: "II", 3: "III", 4: "IV"}
TILE = 512


def collect_cases(grading_files: list[str], min_gap: int, limit: int) -> list[dict]:
    cases: dict[str, dict] = {}
    for path in grading_files:
        data = json.loads(repo_path(path).read_text(encoding="utf-8"))
        weights = data.get("weights", "")
        for row in data["rows"]:
            gap = row["predicted"] - row["truth"]
            if gap < min_gap:
                continue
            name = row["image"]
            # 同一张图可能被多个模型判高，保留最严重的那条，并记下模型
            if name in cases and cases[name]["gap"] >= gap:
                continue
            cases[name] = {
                "image": name,
                "truth": row["truth"],
                "predicted": row["predicted"],
                "gap": gap,
                "weights": weights,
                "tag": data.get("tag", ""),
            }
    ordered = sorted(cases.values(), key=lambda c: (-c["gap"], c["predicted"], c["image"]))
    return ordered[:limit]


def draw_case(image_path: Path, detections, gt_detections, caption: str) -> Image.Image:
    image = Image.open(image_path).convert("RGB").resize((TILE, TILE))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(11, TILE // 34))

    for detection in gt_detections:
        x0, y0, x1, y1 = detection.box
        draw.rectangle([x0, y0, x1, y1], outline=(48, 209, 88), width=3)
    for detection in detections:
        x0, y0, x1, y1 = detection.box
        draw.rectangle([x0, y0, x1, y1], outline=(255, 59, 48), width=3)
        text = f"{detection.cls_name} {detection.confidence:.2f}"
        draw.text((x0 + 3, max(0, y0 - 13)), text, fill=(255, 59, 48), font=font)

    bar = Image.new("RGB", (TILE, 26), (20, 20, 20))
    ImageDraw.Draw(bar).text((6, 6), caption, fill=(235, 235, 235), font=font)
    canvas = Image.new("RGB", (TILE, TILE + 26))
    canvas.paste(image, (0, 0))
    canvas.paste(bar, (0, TILE))
    return canvas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出判级争议样本复核图")
    parser.add_argument("--grading", nargs="+", required=True, help="grading_*.json 列表")
    parser.add_argument("--dataset", default="configs/dataset.msdd.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="docs/review")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--per-sheet", type=int, default=4, choices=[4, 6])
    parser.add_argument("--min-gap", type=int, default=2, help="判级比参考高多少级才入选")
    args = parser.parse_args(argv)

    cfg = load_dataset(args.dataset)
    root = repo_path(cfg["path"])
    names = cfg["names"]
    image_dir = root / "images" / args.split
    label_dir = root / "labels" / args.split

    cases = collect_cases(args.grading, args.min_gap, args.limit)
    if not cases:
        raise SystemExit("没有符合条件的争议样本")
    print(f"入选 {len(cases)} 张（判级高出参考 ≥{args.min_gap} 级）")

    out_dir = repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    detectors: dict[str, object] = {}
    tiles: list[Image.Image] = []
    rows: list[dict] = []

    for index, case in enumerate(cases, start=1):
        weights = case["weights"]
        if weights not in detectors:
            detectors[weights] = load_detector(weights)
        detector = detectors[weights]

        image_path = image_dir / case["image"]
        image = np.asarray(Image.open(image_path).convert("RGB"))
        height, width = image.shape[:2]
        result = detector.predict(image, conf=0.25)
        gt = label_to_detections(label_dir / f"{image_path.stem}.txt", width, height, names, True)

        area = 0.0
        for detection in result.detections:
            area += detection.area / (width * height)
        caption = (
            f"#{index:02d} 标注 {LEVEL_LABEL[case['truth']]} → 模型 {LEVEL_LABEL[case['predicted']]}"
            f"  面积 {area * 100:.0f}%  框 {len(result.detections)}"
        )
        tiles.append(draw_case(image_path, result.detections, gt, caption))
        rows.append(
            {
                "case_id": f"{index:02d}",
                "sheet": f"sheet_{(index - 1) // args.per_sheet + 1:02d}.png",
                "file": case["image"],
                "truth": LEVEL_LABEL[case["truth"]],
                "predicted": LEVEL_LABEL[case["predicted"]],
                "gap": case["gap"],
                "model": (Path(case["weights"]).parent.parent.name if Path(case["weights"]).parent.name == "weights" else Path(case["weights"]).parent.name),
                "pred_area_pct": round(area * 100, 1),
                "ai_judgement": "",  # 由 AI 复核填写
                "your_judgement": "",  # 留给人填写：模型误检 / 标注漏标 / 存疑
                "note": "",
            }
        )

    cols = 2  # 2x2 或 2x3，保证每格足够大、整图不过分拉长
    per = args.per_sheet
    for sheet_index in range(0, len(tiles), per):
        chunk = tiles[sheet_index : sheet_index + per]
        rows_count = (len(chunk) + cols - 1) // cols
        sheet = Image.new(
            "RGB",
            (cols * TILE + (cols + 1) * 8, rows_count * (TILE + 26) + (rows_count + 1) * 8),
            (32, 32, 32),
        )
        for offset, tile in enumerate(chunk):
            row, col = divmod(offset, cols)
            sheet.paste(tile, (8 + col * (TILE + 8), 8 + row * (TILE + 26 + 8)))
        target = out_dir / f"sheet_{sheet_index // per + 1:02d}.png"
        sheet.save(target, quality=92)
        print(f"  {target.name}  {len(chunk)} 张")

    csv_path = out_dir / "index.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n复核图目录: {out_dir}")
    print(f"待填表格: {csv_path}")
    print("your_judgement 一列填：模型误检 / 标注漏标 / 存疑")
    return 0


if __name__ == "__main__":
    sys.exit(main())
