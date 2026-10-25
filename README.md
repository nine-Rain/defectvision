# DefectVision

**用自己的数据训练一个表面缺陷检测模型，从基线到量化部署，全流程可复现。**

一个能跑通「数据体检 → 基线 → 微调 → 量化导出 → 延迟基准 → Web Demo」的最小骨架。
默认场景是工业表面缺陷（划痕 / 凹陷 / 污渍），改 `configs/dataset.yaml` 就能换成任何检测任务。

<!--
徽章：推到 GitHub 后换成你自己的地址
[![CI](https://github.com/<你的ID>/defectvision/actions/workflows/ci.yml/badge.svg)](https://github.com/<你的ID>/defectvision/actions/workflows/ci.yml)
-->

## 为什么做这个

现成模型在**你的**场景上通常不够用：类别不认识、小目标漏检、光照一变就崩。
但"从头训一个模型"既慢又没必要。这个项目的思路是：

> 先用 COCO 预训练权重拿到零样本基线 → 用自有数据微调 → 量化上端侧 →
> 用同一套验证集把每一步的收益和代价都量出来。

所以它产出的不是"一个模型"，而是**一张能被追问的证据表**。

## Demo

<!-- 录屏后放这里：docs/demo.gif -->
> 待补充：`python -m src.app` 起服务，录 10 秒 GIF 存到 `docs/demo.gif`，把下面这行取消注释。
>
> `![demo](docs/demo.gif)`

## 结果

> ⚠️ 下表是**示例数字**，用来展示表格该长什么样。请替换成 `results/metrics.md` 里你自己跑出来的结果。
> 示例数据是"合成缺陷数据集 + yolo11n"，不代表真实场景表现。

| 方案 | mAP50 | mAP50-95 | 体积 (MB) | 推理 (ms) |
| --- | ---: | ---: | ---: | ---: |
| yolo11n（COCO 权重，零样本） | 0.213 | 0.071 | 5.4 | 34.2 |
| yolo11n（自有数据微调） | 0.781 | 0.512 | 5.4 | 34.0 |
| yolo11n + INT8 | 0.762 | 0.494 | 1.6 | 19.3 |

这张表由命令自动生成，不是手抄的：

```bash
python -m src.evaluate --weights runs/baseline/weights/best.pt --tag baseline --note "COCO 零样本"
python -m src.evaluate --weights runs/finetune/weights/best.pt --tag finetune --note "自有数据微调"
python -m src.evaluate --compare        # 重新生成 results/metrics.md
```

## 快速开始（10 分钟，不需要真实数据）

先把流水线跑通，再谈效果。

```bash
git clone https://github.com/<你的ID>/defectvision.git
cd defectvision

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[train,demo,dev]"
# 如果上面这条卡住（国内网络常见），改用：
#   pip install -r requirements.txt

# 1. 生成 160 张合成缺陷图（临时用，跑完可删）
python scripts/make_demo_dataset.py

# 2. 冒烟训练：1 epoch，CPU 也能跑
python -m src.train --smoke --dataset configs/dataset.demo.yaml

# 3. 评估 → 自动写 results/ 对比表
python -m src.evaluate --weights runs/smoke/weights/best.pt --tag smoke

# 4. 起 Demo
python -m src.app --weights runs/smoke/weights/best.pt
```

打开 <http://127.0.0.1:7860> 上传一张图，能看到框和耗时，说明整条链路是通的。

## 换成你自己的数据

### 1. 采集

目标 **≥ 500 张**，其中至少 10% 是负样本（无缺陷的良品）。
手机拍就够用，但要注意：**训练时的光照、角度、距离要和实际使用一致**。

### 2. 标注

用 [Label Studio](https://labelstud.io/) 或 [Roboflow](https://roboflow.com/)，导出 YOLO 格式。
标注规范见 [docs/labeling.md](docs/labeling.md)——**先读再标**，能省掉一轮返工。

### 3. 摆成这个结构

```
data/dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/     # 与图片同名的 .txt
    ├── val/
    └── test/
```

图片和标签名字必须一一对应。负样本的 `.txt` 留空文件，不要不建。

### 4. 体检（别跳过）

```bash
# 改 configs/dataset.yaml 里的 path 和 names
python -m src.prepare stats     # 标签格式 + 类别分布
python -m src.prepare dups      # 重复图 / 跨 split 泄漏
```

`stats` 会报出：字段数不对、类别 id 越界、坐标没归一化、缺标签。
`dups` 会报出跨 split 的重复图——这是验证指标虚高最常见的元凶。

如果图片和标签是平铺的，先切分：

```bash
python -m src.prepare split --src data/raw --out data/dataset --ratios 0.8 0.1 0.1
```

### 5. 训练

```bash
python -m src.train --dry-run                          # 先看参数
python -m src.train --model yolo11n.pt --name v1        # 基线
python -m src.train --model yolo11s.pt --name v2        # 换大模型对比
```

## 命令速查

| 目的 | 命令 |
| --- | --- |
| 数据体检 | `python -m src.prepare stats` |
| 查重复/泄漏 | `python -m src.prepare dups` |
| 切分数据 | `python -m src.prepare split --src data/raw --out data/dataset` |
| 冒烟训练 | `python -m src.train --smoke` |
| 正式训练 | `python -m src.train --name baseline` |
| 续训 | `python -m src.train --resume runs/baseline/weights/last.pt` |
| 评估 | `python -m src.evaluate --weights <path> --tag <name>` |
| 重建对比表 | `python -m src.evaluate --compare` |
| 导出 ONNX | `python -m src.export --weights <path>` |
| INT8 量化 | `python -m src.export --weights <path> --int8` |
| 延迟基准 | `python -m src.benchmark --weights models/best_int8.onnx` |
| 启动 Demo | `python -m src.app --weights <path>` |

## 项目结构

```
defectvision/
├── configs/
│   ├── dataset.yaml        # 数据路径 + 类别表（改这里换业务）
│   ├── dataset.demo.yaml   # 合成数据配置
│   └── train.yaml          # 超参（改这里换模型）
├── src/
│   ├── config.py           # 配置加载与校验
│   ├── prepare.py          # 数据体检 / 去重 / 切分
│   ├── train.py            # 训练
│   ├── evaluate.py         # 评估 + 自动汇总对比表
│   ├── export.py           # ONNX / INT8 导出
│   ├── benchmark.py        # p50/p95 延迟与体积
│   └── app.py              # Gradio Demo
├── scripts/
│   └── make_demo_dataset.py
├── tests/                  # 不需要 GPU / torch 就能跑
├── docs/
│   ├── labeling.md         # 标注规范
│   ├── roadmap.md          # 六周计划 + 面试自检
│   └── publish.md          # 推到 GitHub 的步骤
├── results/                # 指标 json + metrics.md（提交，体积小）
└── .github/workflows/ci.yml
```

## 设计取舍

- **单一配置源**：所有超参在 YAML，命令行只做覆盖。避免"上次那个参数是多少来着"。
- **体重与产物分离**：`runs/` 和 `data/` 不进 git，`results/*.json` 进 git。指标要能被 review。
- **评估可复现**：每次评估落一个 json，表格由脚本生成。手抄的表格一定会有笔误。
- **测试不依赖 torch**：CI 只装 core + dev 依赖，30 秒跑完。

## 失败案例（模板，请替换成你的）

> 这一节是 README 里最容易被跳过、也最能体现工程判断的部分。

| 现象 | 猜测 | 验证 | 结论 |
| --- | --- | --- | --- |
| 验证集 mAP 0.9，实际用起来很差 | 数据泄漏 | 跑 `prepare dups`，发现 30 组跨 split 重复 | 重新切分，mAP 回落到 0.72，才是真实水平 |
| 小目标全漏 | 输入分辨率不够 | imgsz 640→960 重训 | mAP50 +4.1，延迟 +60%，不值得 |
| 换批次后误检暴增 | 光照分布偏移 | 按批次统计亮度直方图 | 补了 200 张暗光样本，误检减半 |

## 常见问题

**Q：必须用 GPU 吗？**
冒烟和一个小的检测模型在 CPU 上可以跑（慢）。正式训练建议至少 8GB 显存的卡，或用 Colab / AutoDL。

**Q：能改成分割吗？**
可以。`--model yolo11n-seg.pt`，标签换成多边形格式即可，评估脚本会自动匹配 `(M)` 指标。

**Q：数据能放仓库里吗？**
别。图片放 Hugging Face Datasets 或网盘，仓库里只放下载脚本和 md5 校验。

**Q：为什么不用 MMDetection / Detectron2？**
可以用，但本项目的目标是"两周内出可用 demo + 可讲的指标"，Ultralytics 的工程链路最短。换框架是加分项，不是起点。

## License

[MIT](LICENSE)
