# 环境问题对照表

这些不是"理论上可能"，是我在这台机器上真实踩过并修好的。按症状查。

## 1. `pip install` 卡住不动，CPU 却占满

**症状**：装 ultralytics 时卡在 `Requirement already satisfied: numpy`，`Get-Process python`
显示 CPU 一直 100%，网络却是通的。

**原因**：pip 的依赖解析器在版本冲突时疯狂回溯（numpy 太新，某些包没有对应 wheel）。
不是网速问题，等再久也不会好。

**处理**：换 [uv](https://github.com/astral-sh/uv)，它用 Rust 写的解析器，通常几秒完成。

```powershell
# 没有 uv 时，可以像下面这样从 PyPI 直接拿一个（不依赖 pip）
python -c "import io,json,urllib.request,zipfile; m=json.loads(urllib.request.urlopen('https://pypi.org/pypi/uv/json').read()); u=[x for x in m['urls'] if 'win_amd64' in x['filename']][0]; open('uv.zip','wb').write(urllib.request.urlopen(u['url']).read()); zipfile.ZipFile('uv.zip').extractall('tools')"

.\tools\uv-*.data\scripts\uv.exe pip install ultralytics --python .\.venv\Scripts\python.exe --cache-dir .uv-cache
```

> 如果 uv 报 `Could not acquire lock ... uv\cache\.lock: 拒绝访问`，
> 加 `--cache-dir .uv-cache` 绕开默认缓存目录。

## 2. 下载权重报 SSL 证书错误

**症状**：

```
SSLCertVerificationError: unable to get local issuer certificate
Download failure for https://github.com/ultralytics/assets/releases/download/.../yolo11n.pt
```

即使设了 `SSL_CERT_FILE` 指向 certifi 也没用——说明链路上有中间人，证书不是公共 CA 签的。

**处理**：从 Hugging Face 镜像拿官方权重，绕开 GitHub。

```powershell
python scripts/fetch_weights.py --model yolo11n.pt yolo11n-seg.pt
```

下载后放在仓库根目录，Ultralytics 会优先用本地文件，不再联网。

## 3. 训练开始前提示 `AMP: checks skipped`

不是错误。Ultralytics 想下 `yolo26n.pt` 做 AMP 自检，被上面的证书问题挡住了，
于是跳过自检、直接开 AMP。只要 loss 正常下降、mAP 不为 0，就没事。
真遇到 NaN 或 mAP 恒为 0，再加 `amp=False` 重训。

## 4. `Dataset '...' images not found`，路径里多了一层 `datasets`

**症状**：报错说找不到 `...\projects\datasets\data\dataset\images\val`。

**原因**：Ultralytics 解析 dataset.yaml 里的相对 `path` 时，是拼到它自己的
`settings.datasets_dir` 上的，不是相对配置文件。

**处理**：本项目已经在 `src/config.py` 的 `materialize_dataset()` 里统一转成绝对路径，
正常不会遇到。如果你自己写了脚本直接调 `YOLO(...).train(data="configs/xxx.yaml")`，
要么给 `path` 写绝对路径，要么先调用 `materialize_dataset()`。

## 5. 数据集下载报 `HTTP 429 Too Many Requests`

镜像有并发限制。脚本已经做了指数退避，但并发太高仍会失败。

```powershell
python scripts/fetch_dataset.py --workers 4     # 失败后重跑即可，已下载的会跳过
```

## 6. `torch.cuda.is_available()` 是 False

装成了 CPU 版。重装指定 CUDA 源：

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

验证必须输出 `True`：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 7. 训练时显存不够（`CUDA out of memory`）

按这个顺序降：`--batch 8` → `--imgsz 416` → `--model yolo11n.pt`。
4060 8G 跑 yolo11n @ 512 / batch 16 大约占 1.5G，余量很大；
换 yolo11s 或上到 640 时才需要担心。
