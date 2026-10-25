"""DefectVision：自定义表面缺陷检测的完整流水线。"""

from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0"

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Ultralytics 默认把 settings.json 写到当前目录的 Ultralytics/ 下。
# 这个目录名和包名 ultralytics 只差大小写，Windows 上会让 ruff 把它误判为本地模块，
# 导致"本地 lint 通过、CI 挂掉"。统一挪进 runs/。
os.environ.setdefault("YOLO_CONFIG_DIR", str(_REPO_ROOT / "runs" / "ultralytics"))
