# 论文对照表：流程图 → 代码

这份文档把开题/论文里的那张系统流程图，逐块映射到仓库里的实现，
方便写论文时直接引用模块名和文件路径，也方便答辩时现场打开对应代码。

## 一、流程图 → 代码

| 流程图模块 | 仓库里的实现 | 说明 |
| --- | --- | --- |
| ① 数据层 · 数据采集 | `scripts/fetch_dataset.py` | 从 HF 镜像拉 MSDD / NEU-DET，断点续传、429 退避 |
| ① 数据层 · 缺陷标注 | `docs/labeling.md` | 标注规范、一致性检查、常见坑 |
| ① 数据层 · 预处理与增强 | Ultralytics 内置增强；CNN 路线见 `src/data/yolo_dataset.py` | 翻转/裁剪/色彩抖动由框架完成 |
| ① 数据层 · 数据集划分 | `src/prepare.py` + 各数据集自带的 split | `dups` 子命令专治跨 split 泄漏 |
| ② 改进 CNN · ResNet 骨干 | `src/models/detector.py::build_backbone` | ResNet50，取 layer1~layer4（C2~C5） |
| ② 改进 CNN · FPN 多尺度融合 | 同上（`torchvision` 的 `BackboneWithFPN`） | 自顶向下 + 3x3 平滑，统一 256 通道，5 个尺度 |
| ② 改进 CNN · CBAM 注意力 | `src/models/cbam.py` | 通道注意力 + 空间注意力，逐层插在 FPN 输出后 |
| ② 改进 CNN · 检测与分类 | `FasterRCNN` 头 | RPN 出候选框，ROIHead 做分类 + 回归 |
| ③ 缺陷等级判定 | `src/grade.py` + `configs/grading.yaml` | 四项特征加权 + 一票否决，输出 I~IV 级 |
| ④ 系统应用层 | `src/ui/app.py` | PyQt5：导入、检测、判级、可视化、报告、批量 |
| ⑤ 实验验证层 | `results/metrics.md`、`results/*.json` | 指标由脚本生成，不手抄 |
| ⑥ 成果输出 | 本仓库 + 论文 | 代码、模型权重、实验记录都在版本控制里 |

## 二、模型结构（写进论文的正文）

```
输入 512x512x3
  └─ ResNet50
       ├─ layer1 → C2  256ch  stride 4
       ├─ layer2 → C3  512ch  stride 8
       ├─ layer3 → C4 1024ch  stride 16
       └─ layer4 → C5 2048ch  stride 32
  └─ FPN：C5 上采样逐级相加 → P2..P6（统一 256ch）
  └─ CBAM ×5：每个尺度独立一组（通道注意力 + 空间注意力）
  └─ RPN：每个尺度 3 种长宽比，生成候选框
  └─ ROI Align (7x7) → 分类头 + 回归头
输出：类别 + 边界框（可扩展为掩码）
```

参数量约 41.4M（ResNet50-FPN-FasterRCNN + CBAM），可在 `python -m src.train_cnn` 启动日志里看到。

**为什么这么选**（答辩会被问）：
- ResNet50 有 ImageNet 预训练且特征层次清晰，适合 200~512 的中小图；
- FPN 解决"缺陷尺度差异大"——划痕细长、氧化成片；
- CBAM 的通道注意力筛"哪些特征通道对缺陷有用"，空间注意力压背景纹理（金属拉丝纹理极易被误检）；
- 检测头用成熟实现，把变量控制在 backbone 上，消融实验才说得清。

## 三、实验设计

### 数据集

| 数据集 | 规模 | 类别 | 用途 |
| --- | --- | --- | --- |
| MSDD（金属零件） | 1,195 / 150 / 150 | 6 | 主实验、判级演示 |
| NEU-DET（热轧钢带） | 1,440 / 180 / 180 | 6 | 泛化性验证（脚本已就绪，见下） |

> NEU-DET 的接入命令：`python scripts/fetch_dataset.py --repo KeenForgeAI/NEU-DET-corrected --out data/neu-det --slug neu-det`
> 拉取时如果镜像限流（HTTP 429），把 `--workers` 降到 4~6 重跑即可，脚本支持断点续传。
> 本机实测镜像对这份数据限流较严，建议换网络时段或改用官方源 `--endpoint https://huggingface.co`。

两份数据的缺陷形态差异很大（圆件 vs 钢板），跨数据集结果能支撑"方法可迁移"的结论。

### 对比模型（按要求：不与经典 CNN 分类器对比）

| 模型 | 类型 | 备注 |
| --- | --- | --- |
| YOLO11n | 单阶段 | 本项目已有的强基线 |
| YOLOv8s / RT-DETR | 单阶段 / DETR 系 | 可选，补齐"当前较好模型"的覆盖面 |
| ResNet50-FPN（无 CBAM） | 两阶段 | 消融上半 |
| ResNet50-FPN + CBAM（本文） | 两阶段 | 消融下半 |

### 消融实验

固定数据、轮数、输入尺寸，只改 CBAM 开关：

```bash
python -m src.train_cnn --name cnn-nocbam --no-cbam --epochs 20
python -m src.train_cnn --name cnn-cbam   --cbam    --epochs 20
python -m src.evaluate --compare
```

表格由 `results/metrics.md` 自动生成，包含 mAP50、mAP50-95、Recall、参数量、体积。

### 建议写进论文的图表

1. 系统架构图（本文档第二节）
2. 数据集样例 + 标注可视化（`scripts/preview_dataset.py`）
3. 训练损失曲线（`runs/cnn-cbam/history.csv`）
4. PR 曲线 / 混淆矩阵（YOLO 路线由 Ultralytics 自动生成）
5. 消融对比表（`results/metrics.md`）
6. 判级混淆矩阵（人工标注 N 张图的真实等级 vs 系统判定）
7. 桌面端界面截图（`docs/assets/ui.png`）
8. 失败案例图（标签过松、小目标漏检）

## 四、判级规则怎么解释

`configs/grading.yaml` 里四个权重对应四项特征：

| 特征 | 含义 | 归一化上限 |
| --- | --- | --- |
| area_ratio | 缺陷总面积 / 图像面积 | 10% |
| count | 缺陷个数 | 10 |
| severity | 最严重类别的严重度（1~4） | 4 |
| spread | 缺陷外接框覆盖面积占比 | 50% |

得分 = 100 × Σ(权重 × 归一化特征)，再按 25/50/75 切四档。
另有"一票否决"：fold 这类致命缺陷只要面积超过 0.5%，直接判 III 级以上。

**这套规则是可解释的**，答辩时可以现场改 `weights` 演示等级变化，
这比"再训一个分类头"更能说明工程思维。

## 五、局限（论文的"不足与展望"可以直接用）

1. 标签质量是上限：MSDD 的框普遍偏松，mAP50-95 上不去；
2. 两阶段模型推理慢于 YOLO，实时性要求高的产线需要换单阶段 + 量化；
3. 判级阈值是经验规则，没有用工艺数据做标定；
4. 只在两个公开数据集上验证，缺少真实产线的长期数据。
