# Houston Dataset Directory / Houston 数据集目录说明

Please place the following `.mat` files into this directory (`datasets/Houston/`):

请将以下 4 个文件放置在本文件夹中：

1. `Houston13.mat`:
   - Contains variable `'ori_data'`, shape `[210, 954, 48]`.
   - Source domain hyperspectral image (Houston 2013).
2. `Houston13_7gt.mat`:
   - Contains variable `'map'`, shape `[210, 954]`.
   - Source domain ground truth for the 6 shared known classes (values 1 to 6, 0 for background).
3. `Houston18.mat`:
   - Contains variable `'ori_data'`, shape `[210, 954, 48]`.
   - Target domain hyperspectral image (Houston 2018).
4. `Houston18_7gt.mat`:
   - Contains variable `'map'`, shape `[210, 954]`.
   - Target domain open-set ground truth (6 known classes + class 7 mapped to unknown/private class).
