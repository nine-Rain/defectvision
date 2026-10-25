"""从 Hugging Face 下载 YOLO 格式的缺陷数据集。

默认下 MSDD（金属零件，6 类）；也支持任意 HF 上的 YOLO 数据集：

    python scripts/fetch_dataset.py
    python scripts/fetch_dataset.py --repo KeenForgeAI/NEU-DET-corrected --out data/neu-det --slug neu-det

不依赖 huggingface_hub：直接走公开的 HTTP 接口逐文件下载，
这样在受限网络、或者别人 clone 你的仓库之后，都能用一条命令复现。

镜像会限流（HTTP 429），所以做了指数退避 + 失败重试 + 断点续传：
中断了重新跑一遍即可，已下载的文件会跳过。

数据来源与许可写进 <out>/SOURCE.md，请保留署名。
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

DEFAULT_REPO = "rltjr/metal-surface-defect-v1"
DEFAULT_OUT = "data/msdd"
DEFAULT_SLUG = "msdd"
# 实测（国内网络）：官方源单文件约 12s，hf-mirror 约 3.4s，所以默认走镜像。
# 镜像只做转发，内容与官方一致；想用官方源加 --endpoint https://huggingface.co
DEFAULT_ENDPOINT = "https://hf-mirror.com"
META_FILES = ("data.yaml", "meta.csv", "README.md", "classes.txt", "LICENSE")

FALLBACK_NAMES = {
    0: "dark_spot",
    1: "fold",
    2: "oxidation",
    3: "porosity",
    4: "scratch",
    5: "stain",
}
UA = {"User-Agent": "defectvision/0.1 (+https://github.com/nine-Rain/defectvision)"}

SOURCE_TEMPLATE = """# 数据来源

- 数据集：{repo}
- 下载地址：<https://huggingface.co/datasets/{repo}>
- 下载工具：`scripts/fetch_dataset.py`（本项目自带，断点续传）
- 获取时间：{date}

使用前请自行确认上游许可（见同目录 README.md / LICENSE），并在论文或产品中署名。
本文件由脚本生成，记录了数据出处，请勿删除。
"""


def _get_json(url: str, retries: int = 3) -> dict | list:
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
    return {}


def list_files(repo: str) -> list[str]:
    """从数据集元信息里拿完整文件列表。

    注意：不要用 /tree/main?recursive=true，那个接口每页最多 1000 条，
    2600 个文件会被静默截断（踩过：train 只下到 847 张）。
    """
    info = _get_json(f"https://huggingface.co/api/datasets/{repo}")
    siblings = info.get("siblings", []) if isinstance(info, dict) else []
    names = [entry["rfilename"] for entry in siblings]
    wanted = [n for n in names if n.startswith(("images/", "labels/")) or n in META_FILES]
    if not wanted:
        raise SystemExit(f"{repo} 里没找到 images/ 或 labels/ 目录，可能不是 YOLO 格式")
    return wanted


def download_one(rel_path: str, out_root: Path, repo: str, endpoint: str, retries: int = 5) -> tuple[str, int]:
    target = out_root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return rel_path, 0  # 断点续传：已存在就跳过

    url = f"{endpoint}/datasets/{repo}/resolve/main/{rel_path}"
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
            time.sleep(min(2 ** attempt * 2, 30))
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(min(2 ** attempt * 2, 30))
    raise RuntimeError(f"下载失败 {rel_path}: {last_error}")


def read_names(out_root: Path, fallback: dict[int, str]) -> dict[int, str]:
    """优先读下载下来的 data.yaml，拿不到再用兜底类别表。"""
    import yaml

    candidate = out_root / "data.yaml"
    if candidate.exists():
        try:
            data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            names = data.get("names")
            if isinstance(names, list):
                return {i: str(n) for i, n in enumerate(names)}
            if isinstance(names, dict):
                return {int(k): str(v) for k, v in names.items()}
        except Exception:  # noqa: BLE001 - 解析失败就退回兜底
            pass
    return fallback


def write_dataset_yaml(out_root: Path, names: dict[int, str], slug: str) -> Path:
    import yaml

    payload = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    target = repo_path("configs") / f"dataset.{slug}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        f"# 由 scripts/fetch_dataset.py 生成，指向 {slug} 数据集\n"
        + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载 HF 上的 YOLO 缺陷数据集")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="HF 数据集 id")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--slug", default=DEFAULT_SLUG, help="生成 configs/dataset.<slug>.yaml")
    parser.add_argument("--workers", type=int, default=12, help="并发数，太高会被镜像限流")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="下载源，默认 hf-mirror.com")
    parser.add_argument("--skip-meta", action="store_true", help="不下载 README 等元数据")
    args = parser.parse_args(argv)

    out_root = repo_path(args.out)
    print(f"数据集: {args.repo}")
    print(f"目标目录: {out_root}")

    files = list_files(args.repo)
    print(f"  待检查 {len(files)} 个文件（images + labels + 元数据）")

    total_bytes = 0
    done = 0
    failed: list[str] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, path, out_root, args.repo, args.endpoint): path
            for path in files
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

    names = read_names(out_root, FALLBACK_NAMES)
    source = SOURCE_TEMPLATE.format(
        repo=args.repo, date=time.strftime("%Y-%m-%d")
    )
    (out_root / "SOURCE.md").write_text(source, encoding="utf-8")
    config = write_dataset_yaml(out_root, names, args.slug)

    print(f"完成：{len(files) - len(failed)}/{len(files)} 个文件，新增 {total_bytes / 1e6:.1f} MB")
    print(f"类别: {names}")
    print(f"已生成 {config}")
    if failed:
        print(f"\n{len(failed)} 个文件失败，重跑一次即可续传：")
        for message in failed[:10]:
            print(f"  - {message}")
        return 1
    print(f"下一步：python -m src.prepare stats --config configs/dataset.{args.slug}.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
