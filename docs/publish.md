# 把项目推到 GitHub 之前，先改这几处

这个骨架的代码已经能跑，但"你的项目"还差这几步。按顺序做，20 分钟搞定。

## 1. 改三处身份信息

| 文件 | 改什么 |
| --- | --- |
| `LICENSE` | `<YOUR NAME>` → 你的名字或 GitHub ID |
| `README.md` | 顶部标题、`<你的仓库地址>`、Demo GIF |
| `pyproject.toml` | `name`、`description`（可选） |

## 2. 建仓库并推送

```powershell
cd "D:\deep seek-harness\projects\defectvision"

# 第一次用 git 的话先配置身份（只做一次）
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"

# 骨架已经 git init 并暂存好了，直接提交
git commit -m "chore: 初始化缺陷检测项目骨架"

# 方式 A：装了 GitHub CLI，一条命令建仓并推送
gh repo create defectvision --public --source=. --push

# 方式 B：先在网页上建空仓库，再手动推
git remote add origin https://github.com/<你的ID>/defectvision.git
git branch -M main
git push -u origin main
```

## 3. 打开 Actions

推上去之后去仓库的 **Actions** 标签页，确认 `CI` 跑绿。
第一次跑会装依赖，大概 1 分钟。绿了就把 README 顶部的徽章换成你的：

```markdown
[![CI](https://github.com/<你的ID>/defectvision/actions/workflows/ci.yml/badge.svg)](https://github.com/<你的ID>/defectvision/actions/workflows/ci.yml)
```

## 4. 把 Issues 建起来

`docs/roadmap.md` 里每一行都是一条 issue。用 GitHub CLI 批量建：

```powershell
gh issue create --title "第1周：数据采集与标注规范" --body "见 docs/roadmap.md"
```

或者直接在 Projects 里拉一个看板：`Todo / In Progress / Done`。
有看板的仓库，面试官一眼就能看出你会不会管项目。

## 5. 第一次真实的 commit 长什么样

不要一次提交 3000 行。参考节奏：

```
feat: 支持 YOLO 多边形标签校验
fix: 修复验证集重复图导致的指标虚高
docs: 补充标注规范中的遮挡判定
chore: 把 ultralytics 固定到 8.3.x
```

约定式提交（Conventional Commits）不是硬性要求，但 `feat/fix/docs/chore` 四个前缀够用了。

## 6. 什么不要提交

`.gitignore` 已经挡了 `data/`、`runs/`、`*.pt`、`*.onnx`。
**再检查一遍**：`git status` 里不应该出现任何图片或权重文件。
数据集太大就放 Hugging Face Datasets 或网盘，README 里给下载脚本和校验和。
