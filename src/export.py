"""导出：把 .pt 变成能部署的 ONNX（可量化）。

    python -m src.export --weights runs/baseline/weights/best.pt
    python -m src.export --weights runs/baseline/weights/best.pt --int8
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from src.config import materialize_dataset, repo_path


def export(
    weights: Path,
    fmt: str = "onnx",
    imgsz: int = 640,
    int8: bool = False,
    data: str = "configs/dataset.yaml",
    dynamic: bool = False,
) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    produced = Path(
        model.export(
            format=fmt,
            imgsz=imgsz,
            int8=int8,
            data=str(materialize_dataset(data)),
            dynamic=dynamic,
            simplify=True,
        )
    )

    # 统一收进 models/，避免权重和配置散落在 runs/ 深处
    target_dir = repo_path("models")
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_int8" if int8 else ""
    target = target_dir / f"{weights.stem}{suffix}.{produced.suffix.lstrip('.')}"
    shutil.copy2(produced, target)

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(weights),
        "format": fmt,
        "imgsz": imgsz,
        "int8": int8,
        "dynamic": dynamic,
        "output": str(target),
        "size_mb": round(target.stat().st_size / (1024 * 1024), 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.export", description="导出部署格式")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--format", default="onnx", choices=["onnx", "engine", "openvino", "torchscript"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--int8", action="store_true", help="INT8 量化，需要校准数据（用 dataset.yaml）")
    parser.add_argument("--dynamic", action="store_true", help="动态 batch / 尺寸")
    parser.add_argument("--data", default="configs/dataset.yaml")
    args = parser.parse_args(argv)

    weights = repo_path(args.weights)
    if not weights.exists():
        raise SystemExit(f"找不到权重: {weights}")

    result = export(weights, args.format, args.imgsz, args.int8, args.data, args.dynamic)
    results_dir = repo_path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"export_{weights.stem}{'_int8' if args.int8 else ''}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"导出完成: {result['output']}  ({result['size_mb']} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
