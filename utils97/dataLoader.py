import torch
from torch.utils.data import Subset, DataLoader

from utils97.myDataset import FeatureDataset, AllDataset, LabelDataset
from utils97.splitData import initDataset, getSourceTrainIndex, getTargetIndex
from utils97.utils import applyPCA

'''
    数据加载器 version 2023.11.14
    
'''
def getDataLoader(args, source_info, target_info, collate_fn=None, drop_last=False):
    # 获取填充后的数据和标签，数据格式[B,W,H],对称填充，标签格式[0]背景，[1,2,3,Unknown]
    source_dataset: dict = initDataset(args, source_info, known_classes=args.source_known_classes)
    target_dataset: dict = initDataset(args, target_info, known_classes=args.target_known_classes, unknown_classes=args.target_unknown_classes)
    # 获取源域训练样本的索引，源域样本按照arg里面的180进行抽取的
    source_index = getSourceTrainIndex(source_dataset['gt'], args, source_info) # 传入的是标签数据
    # 获取全部的目标域样本、已知类样本、未知类样本，目标域的全部样本参与训练。
    target_index: dict = getTargetIndex(target_dataset['gt'], args, target_info)

    # source_dataset['data'],pca = applyPCA(source_dataset['data'], numComponents=3)
    # target_dataset['data'], pca = applyPCA(source_dataset['data'], numComponents=3)
  
    #获取所有数据的dataset，每一个样本是一个patch
    source_all_dataset = AllDataset(source_dataset['data'], args, source_info)
    target_all_dataset = AllDataset(target_dataset['data'], args, target_info)

    source_label_dataset = LabelDataset(source_dataset['data'], source_dataset['gt'], args, source_info)
    target_label_dataset = LabelDataset(target_dataset['data'], target_dataset['gt'], args, target_info)

    source_train_dataset = Subset(source_label_dataset, source_index)
    target_foreground_dataset = Subset(target_label_dataset, target_index['all_index_list'])
    target_known_dataset = Subset(target_label_dataset, target_index['known_index_list'])
    target_unknown_dataset = Subset(target_label_dataset, target_index['unknown_index_list'])

    return {
        'source': {
            'all': DataLoader(source_all_dataset, batch_size=args.batch, shuffle=False),
            'train': DataLoader(source_train_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, drop_last=drop_last)
        },
        'target': {
            'all': DataLoader(target_all_dataset, batch_size=args.batch, shuffle=False),
            'train': DataLoader(target_foreground_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, drop_last=drop_last),
            'test':  DataLoader(target_foreground_dataset, batch_size=getattr(args, 'test_batch_size', 256), shuffle=False, collate_fn=collate_fn),
            'known': DataLoader(target_known_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn),
            'unknown': DataLoader(target_unknown_dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
        }
    }