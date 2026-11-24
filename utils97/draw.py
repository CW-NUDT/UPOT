import os
import warnings

import numpy as np
import torch
from typing import List

# Prefer image IO without GUI; fall back to matplotlib Agg if imageio is unavailable
try:
    import imageio.v2 as imageio  # type: ignore
    _USE_IMAGEIO = True
except Exception:
    import matplotlib
    matplotlib.use('Agg')  # non-GUI backend
    import matplotlib.pyplot as plt
    _USE_IMAGEIO = False

'''
    画分类图 version 2024.05.29
    入口: drawPredictionMap
'''

def getColors():
    return np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255],
            [176, 48, 96],[255,255,255],
            [46, 139, 87], [160, 32, 240], [255, 127, 80], [127, 255, 212],
            [218, 112, 214], [160, 82, 45], [127, 255, 0], [216, 191, 216], [128, 0, 0], [0, 128, 0],
            [0, 0, 128]])

def getClassificationMap(label, unknown=[], unknown_color=[255, 255, 255]):
    colors = getColors()
    image = np.zeros((*label.shape, 3), dtype='uint8')
    for cls in range(1, label.max() + 1):
        image[np.where(label == cls)] = colors[cls - 1]

    for cls in unknown:
        image[np.where(label == cls)] = unknown_color

    return image


def clearBackground(info, image, known_classes=None, unknown_classes=None):
    '''
        version 2024.04.12
        重新排列gt
        使用原始gt时，如果有某些类同时不属于known和unknown，会被画出来
    '''
    from utils97.splitData import transformGT

    gt = transformGT(None, info, known_classes, unknown_classes)
    image[np.where(gt == 0)] = [0, 0, 0]

    return image


def parsePredictionLabel(label: List[torch.Tensor], H):
    '''
        version 2024.04.12
    '''
    label = torch.cat(label).cpu() + 1
    return label.reshape(H, -1)


def drawPredictionMap(label: List[torch.Tensor], name, info, known_classes=[], unknown_classes=[], draw_background=True):
    '''
        version 2024.04.12
    '''
    label = parsePredictionLabel(label, info.image_width)
    image = getClassificationMap(label, unknown=[len(known_classes) + 1])

    if draw_background is False:
        image = clearBackground(info, image, known_classes, unknown_classes)
    saveImage(image, name, 'map')


def saveImage(image, name, path='map'):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    # 自动追加 seed 标识，避免重复追加
    try:
        _seed = os.environ.get('CURRENT_SEED', '').strip()
    except Exception:
        _seed = ''
    final_name = name
    if _seed and f"seed={_seed}" not in name:
        # 若 name 已经包含其它 seed=xxx，不再叠加，保持幂等
        if 'seed=' not in name:
            final_name = f"{name} seed={_seed}"

    out_path = f'{os.path.join(path, final_name)}.png'
    if _USE_IMAGEIO:
        imageio.imwrite(out_path, image)
    else:
        plt.imsave(out_path, image)


def saveFig(filename, path='map', dpi=300):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    # No figure context here; keep as no-op to avoid GUI deps
    pass
