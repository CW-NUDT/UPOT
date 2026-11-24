import json
import torch
import random
import os
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA

from .pyExt import Dict2Obj

def get_datasets_dir():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(repo_root, 'datasets'),
        os.path.abspath('datasets'),
        os.path.join(os.getcwd(), 'datasets')
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.exists(os.path.join(c, 'dataset_config.json')):
            return c
    return os.path.join(repo_root, 'datasets')

def getDatasetInfo(dataset):
    config_path = os.path.join(get_datasets_dir(), "dataset_config.json")
    with open(config_path, "r", encoding="UTF-8") as f:
        info = json.load(f)[dataset]

    return Dict2Obj(info)

def getDataByInfo(info):
    dataset_path = os.path.join(get_datasets_dir(), info.path)

    if info.type is None:
        data = loadmat(os.path.join(dataset_path, info.file_name))[info.mat_name].astype(np.float32)
    elif info.type == 'npy':
        data = np.load(os.path.join(dataset_path, info.file_name)).astype(np.float32)

    return data

def getGTByInfo(info):
    dataset_path = os.path.join(get_datasets_dir(), info.path)

    if info.type is None:
        gt = loadmat(os.path.join(dataset_path, info.gt_file_name))[info.gt_mat_name].astype(np.int64)
    elif info.type == 'npy':
        gt = np.load(os.path.join(dataset_path, info.gt_file_name)).astype(np.int64)
    
    return gt

def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	# 记录当前 seed，供保存图片时自动追加文件名标识
	os.environ['CURRENT_SEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def getDevice(device=None):
    if device is None:
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    elif device == -1:
        return torch.device('cpu')
    else:
        return torch.device(f'cuda:{device}')
def applyPCA(X, numComponents=3):
    X = X.permute(1, 2, 0)
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    newX = torch.tensor(newX)
    newX = newX.permute(2, 0, 1)
    newX = newX.float()
    return newX, pca