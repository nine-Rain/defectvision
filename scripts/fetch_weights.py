"""下载 Ultralytics 预训练权重。

官方权重托管在 GitHub Release，国内网络经常因为证书链或连通性失败：
    SSLCertVerificationError: unable to get local issuer certificate

Ultralytics 同时在 Hugging Face 上维护了官方仓库，内容一致。这里默认从镜像取：

    python scripts/fetch_weights.py                    # yolo11n.pt
    python scripts/fetch_weights.py --model yolo11s.pt yolo11n-seg.pt
    python scripts/fetch_weights.py --endpoint https://huggingface.co
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import repo_path  # noqa: E402

HF_REPO = "Ultralytics/YOLO11"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
UA = {"User-Agent": "defectvision/0.1 (+https://github.com/nine-Rain/defectvision)"}


def fetch(name: str, endpoint: str, out_dir: Path, retries: int = 3) -> Path:
    target = out_dir / name
    if target.exists() and target.stat().st_size > 1_000_000:
        print(f"  {name} 已存在，跳过")
        return target

    url = f"{endpoint}/{HF_REPO}/resolve/main/{name}"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            started = time.time()
            request = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            target.write_bytes(data)
            print(f"  {name}: {len(data) / 1e6:.1f} MB in {time.time() - started:.1f}s")
            return target
        except Exception as exc:  # noqa: BLE001 - 重试所有网络异常
            last_error = exc
            time.sleep(2 ** attempt)
    raise SystemExit(f"下载失败 {name}: {last_error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载 YOLO 预训练权重")
    parser.add_argument(
        "--model",
        nargs="+",
        default=["yolo11n.pt"],
        help="权重文件名，可多个：yolo11n.pt yolo11s.pt yolo11n-seg.pt",
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="默认 hf-mirror.com")
    parser.add_argument("--out", default=".", help="保存目录，默认仓库根目录")
    args = parser.parse_args(argv)

    out_dir = repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"下载到 {out_dir}")
    for name in args.model:
        fetch(name, args.endpoint, out_dir)
    print("完成。Ultralytics 会优先使用当前目录下的同版权重，不会再联网。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
