import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torchmetrics import F1Score
import json
from typing import List, Dict, Any

from .logger import saveJSONFile
# Fallback typing alias for MatrixSequence
try:
    from utils97.typing import MatrixSequence  # type: ignore
except Exception:
    from typing import Sequence, Union
    MatrixSequence = Union[Sequence, np.ndarray, torch.Tensor]

# dictTensorItem is needed for converting tensors to serializable types
try:
    from utils97.pyExt import dictTensorItem  # type: ignore
except Exception:
    def dictTensorItem(dic: dict) -> dict:
        out = {}
        for k, v in dic.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    out[k] = v.detach().cpu().item()
                else:
                    out[k] = v.detach().cpu().tolist()
            else:
                try:
                    out[k] = float(v)
                except Exception:
                    out[k] = v
        return out
from sklearn.metrics import cohen_kappa_score


class AverageMeter:
    def __init__(self):
        self.reset()

    def __str__(self):
        return '%f' % self.value()

    def reset(self):
        self.sum = 0
        self.count = 0

    def add(self, value, count=1):
        self.sum += value
        self.count += count

    def value(self):
        return self.sum / self.count

class ConfusionMeter:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes))

    def add(self, output, target):
        pred = output.max(1)[1].cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        matrix = confusion_matrix(target, pred, labels=np.arange(self.num_classes))
        self.matrix += matrix

    def value(self):
        return self.getOA(), self.getAA()

    def __str__(self):
        return "OA: %.2f%%\nAA: %.2f%%\nClass: %s" % \
               (self.getOA()*100, self.getAA()*100, np.around(self.getClassesAccuracy()*100, 2))

    def getOA(self):
        return np.diag(self.matrix).sum() / self.matrix.sum()

    def getAA(self):
        AA_matrix = self.matrix / self.matrix.sum(1, keepdims=True)
        AA = np.diag(AA_matrix).sum() / self.num_classes
        return AA

    def getClassesAccuracy(self):
        return np.diag(self.matrix / self.matrix.sum(1, keepdims=True))
    
class F1ScoreMeter:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.prediction = []
        self.target = []

    def add(self, prediction, target):
        self.prediction.extend(prediction.cpu().detach().tolist())
        self.target.extend(target.cpu().detach().tolist())

    def value(self, average):
        # ['micro', 'macro']
        f1 = F1Score(self.num_classes, average=average)
        return f1(torch.tensor(self.prediction), torch.tensor(self.target))

class FeatureAndGTGather:
    def __init__(self):
        self.reset()

    def reset(self):
        self.feature_list = []
        self.gt_list = []

    def add(self, feature, gt):
        self.feature_list.append(feature.cpu().detach().numpy())
        self.gt_list.append(gt.cpu().detach().numpy())

    def getFeatureAndGT(self):
        return np.concatenate(self.feature_list, 0), np.concatenate(self.gt_list)
    
def computeOpensetDomainResult(prediction: MatrixSequence, label: MatrixSequence, known_num_classes: int):
    '''
        version 2025.5.22
    '''
    from  torchmetrics import Accuracy
    try:
        from torchmetrics.classification import MulticlassAccuracy
    except ImportError:
        try:
            from torchmetrics import MulticlassAccuracy
        except ImportError:
            # 如果都导入失败，使用Accuracy替代
            MulticlassAccuracy = Accuracy

    if not isinstance(prediction, torch.Tensor):
        prediction = torch.tensor(prediction)
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label)

    known_mask = label < known_num_classes
    unknown_mask = label == known_num_classes

    device = label.device

    oa_meter = Accuracy().to(device)
    aa_meter = MulticlassAccuracy(num_classes=known_num_classes + 1, average=None).to(device)
    known_meter = Accuracy().to(device)
    unknown_meter = Accuracy().to(device)

    oa = oa_meter(prediction, label)
    aa = aa_meter(prediction, label).mean()
    classes_acc = aa_meter(prediction, label)
    oa_known = known_meter(prediction[known_mask], label[known_mask])
    aa_known = classes_acc[:-1].mean()
    unknown = unknown_meter(prediction[unknown_mask], label[unknown_mask])
    hos = (2 * aa_known * unknown) / (aa_known + unknown)

    # kappa 计算（整体 cohen kappa）
    kappa = cohen_kappa_score(label.cpu().numpy().tolist(), prediction.cpu().numpy().tolist())

    return dictTensorItem({
        'oa': oa,
        'aa': aa,
        'classes_acc': classes_acc,
        'oa_known': oa_known,
        'aa_known': aa_known,
        'unknown': unknown,
        'hos': hos,
        'kappa': torch.tensor(kappa, device=device, dtype=torch.float32)
    })


class PredictionTargetGather:
    '''
        version 2023.12.31
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.prediction_list = []
        self.target_list = []

    def update(self, prediction, target):
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu()

        self.prediction_list.append(prediction)
        self.target_list.append(target)

    def get(self):
        return torch.cat(self.prediction_list), torch.cat(self.target_list)


class OpensetDomainMetric:
    '''
        version 2023.12.31
    '''
    def __init__(self, known_num_classes, args):

        self.known_num_classes = known_num_classes
        self.save_path = f'logs/{args.log_name}/{args.log_name} {args.source_dataset}-{args.target_dataset} seed={args.seed}.json'

        self.reset()

    def reset(self):
        self.gather = PredictionTargetGather()
        self.save_dict = None

    def update(self, prediction, target):
        self.gather.update(prediction, target)

    def compute(self):
        self.save_dict = computeOpensetDomainResult(*self.gather.get(), self.known_num_classes)
        return self.save_dict
    
    def save(self, a=False):
        saveJSONFile(self.save_path, self.save_dict, a=a)

    def print(self):
        print(json.dumps(self.save_dict, indent=4))

    def finish(self, a=False):
        self.compute()
        self.print()
        self.save(a=a)
        self.reset()


# ========== 新增：跨 seed 聚合与打印 ==========

def _stack_list(values: List[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def aggregate_openset_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """对多次 seed 运行的 openset 指标做均值与标准差聚合。
    期望每个元素包含键：'oa','aa','kappa','oa_known','aa_known','unknown','hos','classes_acc'。
    """
    if len(results) == 0:
        return {}

    keys_scalar = ['oa', 'aa', 'kappa', 'oa_known', 'aa_known', 'unknown', 'hos']

    agg: Dict[str, Any] = {}

    # 标量项
    for k in keys_scalar:
        vals = _stack_list([r.get(k, np.nan) for r in results])
        agg[f'{k}_mean'] = float(np.nanmean(vals))
        agg[f'{k}_std'] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0

    # 每类准确率
    classes_list = [np.asarray(r['classes_acc'], dtype=np.float64) for r in results if 'classes_acc' in r]
    classes_arr = np.stack(classes_list, axis=0)  # (num_seeds, num_classes)
    agg['classes_acc_mean'] = classes_arr.mean(axis=0).tolist()
    agg['classes_acc_std'] = classes_arr.std(axis=0, ddof=1).tolist() if classes_arr.shape[0] > 1 else [0.0] * classes_arr.shape[1]

    return agg


def pretty_print_aggregate(agg: Dict[str, Any], class_names: List[str] = None):
    """在终端打印均值±标准差，包含每类与综合指标。"""
    if not agg:
        print('No aggregated results to show.')
        return

    def f(mean_key):
        std_key = mean_key.replace('_mean', '_std')
        return f"{agg[mean_key]*100:.2f}±{agg[std_key]*100:.2f}"

    # 每类
    ca_mean = agg['classes_acc_mean']
    ca_std = agg['classes_acc_std']
    print('Per-class accuracy (mean±std) [%]:')
    row = []
    for i, (m, s) in enumerate(zip(ca_mean, ca_std)):
        name = class_names[i] if class_names and i < len(class_names) else str(i + 1)
        row.append(f"{name}:{m*100:.2f}±{s*100:.2f}")
    print(' '.join(row))

    # 综合
    print('Overall metrics (mean±std) [%]:')
    for k in ['oa', 'aa', 'kappa', 'oa_known', 'aa_known', 'unknown', 'hos']:
        print(f"- {k.upper()}: {f(k + '_mean')}")

if __name__ == '__main__':
    a = ConfusionMeter(10)
    print(a)
