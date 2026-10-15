# MiniCLIP

基于 PyTorch 的教学型图文双塔对比学习实验。主版本使用冻结的 ResNet18 与多语言 DistilBERT，投影到 256 维后进行 L2 归一化，通过双向交叉熵学习图文对齐。并非 OpenAI 官方 CLIP 实现。

## 文件与运行

- `modelC10.py`：CIFAR-10 主版本，前 5,000 张训练、前 1,000 张训练脚本内评测，默认 10 epochs。
- `testClipC10.py`：加载 `data/miniclip.pth`，官方测试集前 1,000 张、固定 10 类英文候选分类。
- `model.py`：独立的 ResNet50 + 3000 维投影教学版本，使用 `data/clip_toy/images`，A–J 标签，与主版本检查点不兼容。
- `data/hf_cache`：DistilBERT 本地缓存；`data/cifar10`：CIFAR-10；均不提交 Git。

```bash
python testClipC10.py
# 重新训练会覆盖 data/miniclip.pth，请先备份旧检查点
python modelC10.py
# 教学版本也写入相同检查点名，请使用独立副本或先备份
python model.py
```

新副本将教学训练置于 main 保护中，避免 import 自动训练；测试候选由当前 batch 真实标签修正为固定十类。训练默认 CPU，测试自动选择 MPS/CPU。该任务包含训练时见过的类别，不应称为未见类别零样本学习。原始历史全量测试报告为 73.91%，本次未重跑该报告协议，不能将其当作上述 1,000 张命令的实测结果。

## 环境安装

建议 Python 3.11，在项目根目录执行（所有相对路径均以项目根目录为基准）：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

核心版本依据原实验环境整理；并非跨平台锁文件。首次运行可能下载数据与预训练模型，需要网络与足够磁盘空间。原环境为 Apple Silicon，其他平台需选择适配硬件的 PyTorch 安装包。本次拆分未重新训练或验证完整模型指标。

## 本地资源与 GitHub

本地保留本项目对应的数据及权重副本，不引用原 MiniCLIPTest，也不使用符号链接。`.gitignore` 排除数据、权重、运行产物、缓存和 IDE 配置；这些文件不会随普通 Git 提交上传。其他设备需要下载基础模型/数据，并另行获取自有训练检查点。可通过 GitHub Releases 或 Git LFS 管理权重；请确认数据和基础模型的许可后再分发。未擅自添加开源许可证，请在公开前自行选择代码许可证。

上传到新建空仓库时，可在本目录执行：

```bash
git init
git add .
git status --short
# 确认暂存内容后
git commit -m "Initial commit"
git branch -M main
git remote add origin <你的仓库地址>
git push -u origin main
```

数据与权重仅保留在本地，Git 仓库只包含代码、依赖与说明文件。
