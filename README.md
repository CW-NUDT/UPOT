# UPOT: Uncertainty-Aware Partial Optimal Transport for Open-Set Hyperspectral Domain Adaptation

This repository contains the open-source PyTorch implementation of the method described in the IEEE GRSL paper:
**"Uncertainty-Aware Partial Optimal Transport for Open-Set Hyperspectral Domain Adaptation"**.

---

## 1. Overview / 概述

This repository provides an open-set domain adaptation framework for hyperspectral image classification. It addresses the challenge of aligning known classes across domains while rejecting target-private unknown classes to prevent negative transfer.

主要模块说明：
- `osbp_ot_train.py`: 主训练脚本，支持 Pavia (PU -> PC) 与 Houston (HO13 -> HO18) 任务。
- `osbp_hyperspectral_models.py`: 3D-2D 混合残差特征提取骨干网络与分类器结构。
- `osbp_hyperspectral_config.py`: 超参数解析与数据集配置。
- `eval_checkpoint.py`: 独立评估脚本，用于直接加载权重在全量测试集上做前向推理与指标统计。
- `utils97/`: 数据预处理、空间切片、评估指标计算与日志工具。
- `checkpoints/`: 训练权重保存目录（开源版本不包含二进制权重文件）。
- `datasets/`: 数据集配置与组织指引（开源版本已去除二进制数据文件）。

---

## 2. Environment Setup / 环境配置

推荐使用 Conda 创建 Python 3.10 环境：

```bash
conda create -n upot python=3.10 -y
conda activate upot

# 安装 PyTorch (建议配合本机 CUDA 版本)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt
```

---

## 3. Dataset Preparation / 数据集准备

开源代码库中已移除所有原始数据文件（`.mat`）。运行前请按照 `datasets/README.md` 指引放置数据：

```
datasets/
├── dataset_config.json
├── Pavia/
│   ├── paviaU.mat
│   ├── paviaU_7gt.mat
│   ├── paviaC.mat
│   └── paviaC_OS_gt.mat
└── Houston/
    ├── Houston13.mat
    ├── Houston13_7gt.mat
    ├── Houston18.mat
    └── Houston18_7gt.mat
```

具体波段数、空间尺寸与变量名称要求请参阅 [datasets/README.md](datasets/README.md)。

---

## 4. Evaluation / 模型评估

训练完成后，模型权重将保存在 `checkpoints/` 目录下（开源版本不随附公开权重文件）。可通过 `eval_checkpoint.py` 脚本加载指定权重执行全量测试集前向推理与性能测试：

```bash
# 评估 Pavia 任务训练权重 (PU -> PC, 全量 82,181 测试样本)
python eval_checkpoint.py --checkpoint checkpoints/UPOT_PaviaU_7gt-PaviaC_OS_seed_58_best.pth --batch_size 256

# 评估 Houston 任务训练权重 (HO13 -> HO18, 全量 53,200 测试样本)
python eval_checkpoint.py --checkpoint checkpoints/UPOT_Houston13_7gt-Houston18_7gt_seed_1_best.pth --batch_size 256
```

### Benchmark Results / 论文复现结果对比

#### Pavia Benchmark (Table III, PU -> PC):
- **HOS (调和均值)**: **87.70%** (论文: 86.49% ± 2.51%)
- **OS\* (已知类平均精度)**: **88.63%** (论文: 85.47%)
- **Unk (未知类准确率)**: **86.79%** (论文: 87.54%)
- **OA (总体分类准确率)**: **87.65%**

#### Houston Benchmark (Table IV, HO13 -> HO18):
- **HOS (调和均值)**: **67.14%** (论文: 67.98% ± 1.22%)
- **OS\* (已知类平均精度)**: **61.95%** (论文: 63.09%)
- **Unk (未知类准确率)**: **73.29%** (论文: 73.70%)
- **OA (总体分类准确率)**: **47.92%**

---

## 5. Training / 训练流程

如需从头开始训练，可运行以下命令：

```bash
# 训练 Pavia 任务
python osbp_ot_train.py --source_dataset PaviaU_7gt --target_dataset PaviaC_OS --seed 58 --epochs 30

# 训练 Houston 任务 (推荐使用 Seed 1)
python osbp_ot_train.py --source_dataset Houston13_7gt --target_dataset Houston18_7gt --seed 1 --epochs 25
```

---

## 6. Citation / 论文引用

如果本代码库或实验结果对您的研究有所帮助，请引用以下论文：

```bibtex
@ARTICLE{11782405,
  author={Wang, Chen and Li, Jun and Zhang, Wenpeng and Liu, Shuowei and Liu, Zhen},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Uncertainty-Aware Partial Optimal Transport for Open-Set Hyperspectral Domain Adaptation}, 
  year={2026},
  volume={23},
  number={},
  pages={5508905-5508905},
  keywords={Prototypes;Uncertainty;Labeling;Modeling;Hyperspectral imaging;Modules (abstract algebra);Surface plasmon resonance;Learning (artificial intelligence);Image classification;Hyperspectral image (HSI) classification;open-set domain adaptation (OSDA);optimal transport (OT);uncertainty-aware learning},
  doi={10.1109/LGRS.2026.3751317}}
```

---

## 7. License / 开源协议

本项目采用 MIT License 开源许可协议。
