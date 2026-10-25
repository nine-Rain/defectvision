"""下载 MSDD 金属表面缺陷数据集（Hugging Face，CC BY 4.0）。

不依赖 huggingface_hub：直接走公开的 HTTP 接口逐文件下载，
这样在受限网络、或者别人 clone 你的仓库之后，都能用一条命令复现。

    python scripts/fetch_dataset.py
    python scripts/fetch_dataset.py --workers 12 --endpoint https://huggingface.co

镜像会限流（HTTP 429），所以这里做了指数退避 + 失败重试 + 断点续传：
中断了重新跑一遍即可，已下载的文件会跳过。

数据来源与许可：
    原始数据集 Roboflow Universe — msdd/msdd-prompt-learning-qjz4o (CC BY 4.0)
    整理版 https://huggingface.co/datasets/rltjr/metal-surface-defect-v1 (CC BY 4.0)
    使用时请保留本文件与 data/msdd/SOURCE.md 里的署名。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import repo_path  # noqa: E402

HF_REPO = "rltjr/metal-surface-defect-v1"
TREE_API = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main"
# 实测（国内网络）：官方源单文件约 12s，hf-mirror 约 3.4s，所以默认走镜像。
# 镜像只做转发，内容与官方一致；想用官方源加 --endpoint https://huggingface.co
DEFAULT_ENDPOINT = "https://hf-mirror.com"
SINGLE_FILES = ("data.yaml", "meta.csv", "README.md")

NAMES = {0: "dark_spot", 1: "fold", 2: "oxidation", 3: "porosity", 4: "scratch", 5: "stain"}
UA = {"User-Agent": "defectvision/0.1 (+https://github.com/nine-Rain/defectvision)"}

SOURCE_MD = """# 数据来源

- 名称：MSDD（Metal Surface Defect Dataset）清理版
- 原始发布：Roboflow Universe `msdd/msdd-prompt-learning-qjz4o` v9
- 本仓库下载自：<https://huggingface.co/datasets/rltjr/metal-surface-defect-v1>
- 许可：CC BY 4.0，使用需署名
- 内容：1,300 张 512x512 金属零件表面照片，6 类缺陷，YOLO 多边形标签
  - train 1000 / val 150 / test 150，按零件划分（同一零件的多角度只落在一个 split）
  - 244 张为正常样本（空标签文件）

类别：0 dark_spot / 1 fold / 2 oxidation / 3 porosity / 4 scratch / 5 stain
"""


def _get_json(url: str, retries: int = 3) -> list[dict]:
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    return []


def list_files() -> list[str]:
    """从数据集元信息里拿完整文件列表。

    注意：不要用 /tree/main?recursive=true，那个接口每页最多 1000 条，
    2600 个文件会被静默截断（我踩过这个坑：train 只下到 847 张）。
    """
    info = _get_json(f"https://huggingface.co/api/datasets/{HF_REPO}")
    siblings = info.get("siblings", []) if isinstance(info, dict) else []
    names = [entry["rfilename"] for entry in siblings]
    return [n for n in names if n.startswith(("images/", "labels/")) or n in SINGLE_FILES]


def download_one(rel_path: str, out_root: Path, endpoint: str, retries: int = 5) -> tuple[str, int]:
    target = out_root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return rel_path, 0  # 断点续传：已存在就跳过

    url = f"{endpoint}/datasets/{HF_REPO}/resolve/main/{rel_path}"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            partial = target.with_suffix(target.suffix + ".part")
            partial.write_bytes(data)
            partial.replace(target)
            return rel_path, len(data)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 404:
                raise RuntimeError(f"文件不存在（可能改过名）: {rel_path}") from exc
            # 429 限流 / 5xx 服务端抖动：退避后重试
            time.sleep(min(2 ** attempt * 2, 30))
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(min(2 ** attempt * 2, 30))
    raise RuntimeError(f"下载失败 {rel_path}: {last_error}")


def write_dataset_yaml(out_root: Path) -> Path:
    import yaml

    payload = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": NAMES,
    }
    target = repo_path("configs/dataset.msdd.yaml")
    target.write_text(
        "# 由 scripts/fetch_dataset.py 生成，指向 MSDD 数据集（CC BY 4.0）\n"
        + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载 MSDD 金属表面缺陷数据集")
    parser.add_argument("--out", default="data/msdd")
    parser.add_argument("--workers", type=int, default=12, help="并发数，太高会被镜像限流")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="下载源，默认 hf-mirror.com")
    args = parser.parse_args(argv)

    out_root = repo_path(args.out)
    print(f"目标目录: {out_root}")

    files: list[str] = list_files()
    print(f"  待检查 {len(files)} 个文件（images + labels + 3 个元数据）")

    total_bytes = 0
    done = 0
    failed: list[str] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, path, out_root, args.endpoint): path for path in files
        }
        for future in as_completed(futures):
            rel_path = futures[future]
            try:
                _, size = future.result()
                total_bytes += size
            except Exception as exc:  # noqa: BLE001 - 单个文件失败不该中断整批
                failed.append(f"{rel_path}: {exc}")
            done += 1
            if done % 200 == 0 or done == len(files):
                elapsed = max(time.time() - started, 0.1)
                print(
                    f"  {done}/{len(files)}  {total_bytes / 1e6:.1f} MB"
                    f"  ({total_bytes / 1e6 / elapsed:.2f} MB/s)"
                )

    (out_root / "SOURCE.md").write_text(SOURCE_MD, encoding="utf-8")
    config = write_dataset_yaml(out_root)
    print(f"完成：{len(files) - len(failed)}/{len(files)} 个文件，新增 {total_bytes / 1e6:.1f} MB")
    print(f"已生成 {config}")
    if failed:
        print(f"\n{len(failed)} 个文件失败，重跑一次即可续传：")
        for message in failed[:10]:
            print(f"  - {message}")
        return 1
    print("下一步：python -m src.prepare stats --config configs/dataset.msdd.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
