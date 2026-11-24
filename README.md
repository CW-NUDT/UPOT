# UPOT: Uncertainty-Aware Partial Optimal Transport for Open-Set Hyperspectral Domain Adaptation

This repository contains the open-source PyTorch implementation of the method described in the IEEE GRSL paper:
**"Uncertainty-Aware Partial Optimal Transport for Open-Set Hyperspectral Domain Adaptation"**.

---

## 1. Overview / 概述

This repository provides an open-set domain adaptation framework for hyperspectral image classification. It addresses the challenge of aligning known classes across domains while rejecting target-private unknown classes to prevent negative transfer.


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

运行前请按照 `datasets/README.md` 指引放置数据：

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

---


## 4. Citation / 论文引用

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

