# 自建数据集：采集、标注与接入

对应开题报告里"自建缺陷数据集"一项。目标不是凑数量，而是**证明这套流程能迁移到你自己的图**。
60~100 张、2~3 类缺陷就够写进论文的"泛化性验证"一节。

## 一、素材从哪来（按推荐顺序）

| 来源 | 合法性 | 建议 |
| --- | --- | --- |
| **自己拍** | 完全自主 | 首选。手机拍零件划痕/锈斑，拍 60~100 张 |
| **授权素材站**（Pexels / Unsplash） | 平台许可允许免费使用 | 可作补充，**必须记录来源与许可** |
| **视频抽帧** | 取决于视频版权 | 自己拍的视频最稳；他人视频需获授权 |
| 社交媒体/自媒体截图 | 版权不清晰 | **不建议**用于学位论文，答辩被问来源会很难解释 |

论文里必须有一张"数据来源表"：来源、数量、许可、采集时间。`provenance.csv` 就是为这张表准备的。

## 二、图片采集

### 从文件夹（浏览器另存、素材站下载）

```powershell
python scripts/collect_images.py --src D:\downloads\parts --out data/custom_raw
```

会做校验、按内容去重（dHash）、统一命名，并生成 `provenance.csv`。

### 从直链列表

```powershell
# urls.txt 每行一个图片直链
python scripts/collect_images.py --urls urls.txt --out data/custom_raw --max-side 1280
```

### 从视频抽帧

```powershell
python scripts/extract_frames.py --video D:\videos\parts.mp4 --out data/custom_raw --every 0.5
python scripts/extract_frames.py --video-dir D:\videos --out data/custom_raw --every 0.3
```

默认参数：每 0.5 秒抽一帧、短边 <480 丢弃、拉普拉斯方差 <60 判为模糊、dHash 距离 ≤5 判为重复。
实测一段 60 帧、含 8 个不同画面的视频，抽帧后保留 7 张（去掉了 53 张近似帧）。

> **为什么用 dHash 而不是 aHash**：同一个零件的不同角度在 aHash 下距离可能是 0（被误删），
> 在 dHash 下是 14~21，能正确区分。这个坑实测踩过。

## 三、标注

用 [Label Studio](https://labelstud.io/)：

```powershell
pip install label-studio
label-studio start
```

1. 新建项目 → 选 **Object Detection with Bounding Boxes**
2. 类别建议**沿用 MSDD 的六类**（`scratch / stain / oxidation / dark_spot / porosity / fold`），
   不要自造一套新名词——否则跨数据集实验没法做
3. 按 `docs/labeling.md` 的规范标：紧贴外轮廓、遮挡过半不标、短边 <8px 不标
4. 导出的 JSON 保持默认格式即可

**负样本别忘了**：拍 10~15 张**没有缺陷**的正常件，标注里空着。全是缺陷样本会让模型到处误检。

## 四、接入训练

```powershell
# Label Studio 导出的 JSON
python scripts/import_custom_dataset.py --src data/custom_raw --format labelstudio `
    --annotations D:\export\project.json --out data/custom --slug custom

# 已经是 YOLO 格式（图片旁有同名 .txt）
python scripts/import_custom_dataset.py --src data/custom_raw --format yolo `
    --names scratch,stain,oxidation --out data/custom --slug custom
```

脚本会按 8:1:1 切分、把没标注的图写成空标签（负样本）、生成 `configs/dataset.custom.yaml`。

接入前先体检：

```powershell
python -m src.prepare stats --config configs/dataset.custom.yaml
python -m src.prepare dups  --config configs/dataset.custom.yaml
```

## 五、训练与评估

```powershell
python -m src.train --dataset configs/dataset.custom.yaml --imgsz 512 --epochs 50 --name custom-yolo
python -m src.evaluate --weights runs/custom-yolo/weights/best.pt `
    --data configs/dataset.custom.yaml --split test --tag custom-test --note "自建数据"
python -m src.evaluate_grading --weights runs/custom-yolo/weights/best.pt `
    --dataset configs/dataset.custom.yaml --split test --tag custom-grading
```

因为类别是 MSDD 的子集，`configs/grading.yaml` 的规则可以直接复用。

## 六、当前已收集的种子素材

`data/custom_raw/` 下有 11 张来自 Pexels 的金属表面缺陷照片（锈蚀、漆面剥落、划痕），
每张都在 `provenance.csv` 里记录了原始链接与许可（Pexels License，免费使用、需注明来源）。

这批图的作用是**跑通流程**，不足以支撑泛化性结论。要达到论文要求，还需补：

- [ ] 自己拍摄的零件照片 60 张以上（含 10~15 张负样本）
- [ ] 或自己拍摄的视频 2~3 段，抽帧后去重
- [ ] 标注并跑通 `import_custom_dataset.py`
- [ ] 在自建集上训练 + 评估，写进论文"泛化性验证"
