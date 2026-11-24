import os
import logging
import time
import json
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def getLogger(name, args):
    
    '''
    version 2022.11.01
    '''

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    text = 'dataset=%s seed=%d' % (args.dataset, args.seed)
    if args.train_num != 0:
        text += ' train_num=%.2f' % args.train_num
    else:
        text += ' train_rate=%.2f' % args.train_rate

    # 使用FileHandler输出到文件
    if not os.path.exists('log'):
        os.makedirs('log')
    fh = logging.FileHandler('log/%s %s %s.txt' %
                             (name,
                              time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                              text))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # 使用StreamHandler输出到屏幕
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # 添加两个Handler
    logger.addHandler(ch)
    logger.addHandler(fh)

    # 设置信息
    text += ' batch=%d patch=%d epochs=%d' % (args.batch, args.patch, args.epochs)
    logger.info(name)
    logger.info(text)

    return logger

def saveFile(path, content):
    file_path, file_name = os.path.split(path)
    
    if file_path and not os.path.exists(file_path):
        os.makedirs(file_path, exist_ok=True)

    # 原子写入：先写到临时文件，再替换
    tmp_path = f"{path}.tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(content)
    os.replace(tmp_path, path)

def saveJSONFile(path, save_dict, a=False):
    '''
        version 2025.09.18 (hardened)
    '''
    def _default(o):
        # 尝试将 numpy / torch 类型转为可序列化
        try:
            import numpy as _np
            import torch as _torch
        except Exception:
            _np = None
            _torch = None
        if _torch is not None and isinstance(o, _torch.Tensor):
            return o.detach().cpu().tolist() if o.numel() > 1 else o.detach().cpu().item()
        if _np is not None:
            if isinstance(o, _np.generic):
                return o.item()
            if isinstance(o, _np.ndarray):
                return o.tolist()
        try:
            return float(o)
        except Exception:
            return str(o)

    if a:
        # 合并已存在内容，不存在则视为 {}
        prev = {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                prev = json.loads(f.read() or '{}')
        except FileNotFoundError:
            prev = {}
        except json.JSONDecodeError:
            # 文件损坏则忽略旧内容
            prev = {}
        except PermissionError:
            # 权限问题时，退化为直接写入新内容
            prev = {}
        save_dict = {**prev, **save_dict}

    # 确保目录存在 + 原子写入
    content = json.dumps(save_dict, indent=4, ensure_ascii=False, default=_default)
    saveFile(path, content)


class ProgressLogger:

    '''
        version 2023.11.06
    '''

    def __init__(self, epochs: int):

        self.epoch = 0

        self.progress_bar = tqdm(range(1, epochs + 1))
        self.writer = Writer()

    def update(self, dic: dict):
        self.progress_bar.update()
        self.progress_bar.set_description(f'epoch: {self.epoch + 1}')
        self.progress_bar.set_postfix(dic)
        self.writer.update(dic, self.epoch)

        self.epoch += 1

    def add_information(self, dic: dict):
        self.progress_bar.set_postfix(dic)
        self.writer.update(dic, self.epoch)

class Writer:

    '''
        version 2023.10.27
    '''
    
    def __init__(self):
        self.writer = SummaryWriter('logs')

    def update(self, dic: dict, epoch: int):
        for key in dic:
            self.writer.add_scalar(key, dic[key], epoch)
