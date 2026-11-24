# Pavia Dataset Directory / Pavia 数据集目录说明

Please place the following `.mat` files into this directory (`datasets/Pavia/`):

请将以下 4 个文件放置在本文件夹中：

1. `paviaU.mat`:
   - Contains variable `'ori_data'`, shape `[610, 340, 102]`.
   - Source domain hyperspectral image (Pavia University).
2. `paviaU_7gt.mat`:
   - Contains variable `'map'`, shape `[610, 340]`.
   - Source domain ground truth for the 7 shared known classes (values 1 to 7, 0 for background).
3. `paviaC.mat`:
   - Contains variable `'ori_data'`, shape `[1096, 715, 102]`.
   - Target domain hyperspectral image (Pavia Center).
4. `paviaC_OS_gt.mat`:
   - Contains variable `'map'`, shape `[1096, 715]`.
   - Target domain open-set ground truth (7 known classes + class 9 mapped to unknown/private class).
