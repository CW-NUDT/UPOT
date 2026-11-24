# Dataset Preparation Guide / 数据集准备指南

This directory contains the configuration and file structure for the hyperspectral datasets used in the paper:
**"Uncertainty-Aware Partial Optimal Transport for Open-Set Hyperspectral Domain Adaptation" (IEEE GRSL 2026)**.

本目录包含论文中使用的两个开放集高光谱遥感影像域适应基准数据集（**Pavia** 与 **Houston**）的配置文件与组织规范。由于高光谱遥感数据体积较大且受各方原始版权限制，开源代码库不随附二进制数据文件（`.mat`）。请根据以下指引准备并放置数据文件。

---

## 1. Directory Structure / 目录结构规范

将下载或生成的 `.mat` 数据与真值标签文件分别放置在对应的子文件夹中，结构如下：

```
datasets/
├── dataset_config.json          # 数据集读取元信息配置文件
├── README.md                    # 本说明文档
├── Pavia/
│   ├── README.md                # Pavia 任务具体说明
│   ├── paviaU.mat               # [源域数据] Pavia University 影像 (ori_data, 610x340x102)
│   ├── paviaU_7gt.mat           # [源域真值] 7 类已知地物标注 (map, 610x340)
│   ├── paviaC.mat               # [目标域数据] Pavia Center 影像 (ori_data, 1096x715x102)
│   └── paviaC_OS_gt.mat         # [目标域开放集真值] 7 类已知 + 1 类未知 (map, 1096x715)
└── Houston/
    ├── README.md                # Houston 任务具体说明
    ├── Houston13.mat            # [源域数据] Houston 2013 影像 (ori_data, 210x954x48)
    ├── Houston13_7gt.mat        # [源域真值] 6 类已知地物标注 (map, 210x954)
    ├── Houston18.mat            # [目标域数据] Houston 2018 影像 (ori_data, 210x954x48)
    └── Houston18_7gt.mat        # [目标域开放集真值] 6 类已知 + 1 类未知 (map, 210x954)
```

---

## 2. Dataset Details & Specifications / 数据集详细规格

### 2.1 Pavia Benchmark (PU $\to$ PC, Cross-Sensor OSDA)
- **源域 (Source)**: Pavia University (PU), 由 ROSIS 传感器采集于意大利帕维亚大学。空间分辨率 1.3m，波段数 103（剔除 1 个噪声波段后保留 102 波段），尺寸 $610 \times 340$。
- **目标域 (Target)**: Pavia Center (PC), 由 ROSIS 传感器采集于帕维亚市中心。波段数 102，尺寸 $1096 \times 715$。
- **开放集类别设计 (7 个已知类 + 1 个私有未知类)**:
  - **已知类 (7类)**: C1: Tree (树木), C2: Asphalt (沥青道路), C3: Brick (砖块建筑), C4: Bitumen (沥青混合料), C5: Shadow (阴影), C6: Meadows (草地), C7: Bare Soil (裸土)。
  - **目标域私有未知类 (Unk)**: C8: Tiles (瓦片屋顶)。
- **测试样本量**: 目标域前景点共 **82,181** 个像素（已知类 39,355，未知类 42,826）。

### 2.2 Houston Benchmark (HO13 $\to$ HO18, Cross-Temporal OSDA)
- **源域 (Source)**: Houston 2013, 来自 IEEE GRSS Data Fusion Contest (DFC 2013)，由 CASI 传感器采集，尺寸 $210 \times 954 \times 48$。
- **目标域 (Target)**: Houston 2018, 来自 IEEE GRSS Data Fusion Contest (DFC 2018)，经过重采样并保留与 2013 相同波谱范围的 48 个光谱波段。
- **开放集类别设计 (6 个已知类 + 1 个私有未知类)**:
  - **已知类 (6类)**: C1: Grass healthy (健康草地), C2: Grass stressed (受压草地), C3: Trees (树木), C4: Water (水体), C5: Residential buildings (住宅建筑), C6: Non-Residential buildings (非住宅建筑)。
  - **目标域私有未知类 (Unk)**: C7: Road (道路 - 目标域私有地物)。
- **测试样本量**: 目标域前景点共 **53,200** 个像素（已知类 46,835，未知类 6,365）。

---

## 3. Data Format & Variable Names / 数据格式与变量命名要求

代码内部通过 `scipy.io.loadmat` 读取 `.mat` 文件，请确保各文件中字典变量名（Variable Keys）符合规范：
- **高光谱影像数据 (`.mat`)**: 内部变量名需为 `'ori_data'`，数据维度为 `[H, W, C]` (float32 或 double)。
- **真值标签数据 (`.mat`)**: 内部变量名需为 `'map'`，数据维度为 `[H, W]` (int32 或 uint8)，其中 `0` 表示背景无标签像素，`1` 至 `K` 表示各已知类，`K+1` 表示未知类。

---

## 4. How to Obtain Datasets / 如何获取原始数据

1. **Pavia University & Pavia Center**:
   - 可在 [Computational Intelligence Group (UPV/EHU)](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) 免费下载原始 ROSIS Pavia 数据集。
2. **Houston 2013 & Houston 2018**:
   - 可在 [IEEE GRSS IADTC 官网](https://www.grss-ieee.org/community/technical-committees/2013-ieee-grss-data-fusion-contest/) 申请与下载 DFC 2013 与 DFC 2018 原始高光谱数据。
