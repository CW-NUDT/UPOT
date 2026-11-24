#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UPOT 模型权重快速检测与验证脚本
用于快速加载保存的最佳权重文件并在测试集上执行前向推理，计算并展示全部评估指标
"""

import os
import sys
import argparse
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 确保当前工作目录能够找到 datasets 配置
if os.path.exists(os.path.join(current_dir, 'datasets', 'dataset_config.json')):
    os.chdir(current_dir)

import osbp_hyperspectral_models as models
from osbp_hyperspectral_config import parse_args
from utils97.utils import getDatasetInfo, getDevice, seed_torch
from utils97.dataLoader import getDataLoader
from utils97.meter import computeOpensetDomainResult


def evaluate_checkpoint(checkpoint_path, batch_size=64, device_id=0):
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(current_dir, checkpoint_path)
    
    if not os.path.exists(checkpoint_path):
        # 尝试在 checkpoints 目录寻找
        alt_path = os.path.join(current_dir, "checkpoints", os.path.basename(checkpoint_path))
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
        else:
            raise FileNotFoundError(f"找不到权重文件: {checkpoint_path}")

    print("=" * 70)
    print(f"正在加载模型权重文件: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    seed = checkpoint.get('seed', 58)
    source_dataset = checkpoint.get('source_dataset', 'PaviaU_7gt')
    target_dataset = checkpoint.get('target_dataset', 'PaviaC_OS')
    in_channels = checkpoint.get('in_channels', 102)
    num_classes = checkpoint.get('num_classes', 8)
    patch_size = checkpoint.get('patch', 15)
    source_known_classes = checkpoint.get('source_known_classes', [1, 2, 3, 4, 5, 6, 7])
    target_known_classes = checkpoint.get('target_known_classes', [1, 2, 3, 4, 5, 6, 7])
    target_unknown_classes = checkpoint.get('target_unknown_classes', [9])
    saved_epoch = checkpoint.get('epoch', None)
    early_stopped = checkpoint.get('early_stopped', False)

    print(f"权重元信息:")
    print(f"  随机种子 (Seed): {seed}")
    print(f"  源域 -> 目标域: {source_dataset} -> {target_dataset}")
    print(f"  已知类别数: {len(source_known_classes)}, 类别列表: {source_known_classes}")
    print(f"  未知类别映射: {target_unknown_classes} -> 标签 {len(source_known_classes)}")
    print(f"  输入通道数: {in_channels}, Patch 大小: {patch_size}x{patch_size}")
    if saved_epoch is not None:
        print(f"  保存轮次 (Epoch): {saved_epoch} (早停: {early_stopped})")
    print("=" * 70)

    # 设置设备
    device = getDevice(device_id)
    print(f"推理设备: {device} ({torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'})")

    # 构建并加载模型
    model = models.Net(
        in_channels=in_channels,
        num_classes=num_classes,
        patch_size=patch_size
    )
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 构建数据加载参数
    args = argparse.Namespace(
        batch=batch_size,
        patch=patch_size,
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        source_known_classes=source_known_classes,
        target_known_classes=target_known_classes,
        target_unknown_classes=target_unknown_classes,
        seed=seed,
        train_num=180,
        few_train_num=140
    )
    seed_torch(seed)

    source_info = getDatasetInfo(source_dataset)
    target_info = getDatasetInfo(target_dataset)

    print("正在加载测试集数据...")
    loaders = getDataLoader(args, source_info, target_info)
    test_loader = loaders['target']['test']
    total_test_samples = len(test_loader.dataset)
    print(f"测试集总样本数: {total_test_samples}")

    print("正在进行前向推理与评估...")
    start_time = time.time()
    pred_y = []
    true_y = []

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing", leave=True):
            data = data.to(device)
            output = model(data)
            pred = output.max(1)[1]
            pred_y.extend(pred.cpu().tolist())
            true_y.extend(target.tolist())

    elapsed_time = time.time() - start_time
    print(f"推理完成！耗时: {elapsed_time:.2f} 秒 (平均每秒处理 {total_test_samples / elapsed_time:.1f} 个样本)")

    # 计算指标
    known_num = len(source_known_classes)
    res = computeOpensetDomainResult(pred_y, true_y, known_num)

    oa = res['oa'] * 100
    aa = res['aa'] * 100
    os_star = res['aa_known'] * 100  # OS* (AA Known)
    oa_known = res['oa_known'] * 100
    unk_acc = res['unknown'] * 100
    hos = res['hos'] * 100
    kappa = res['kappa'] * 100
    classes_acc = [acc * 100 for acc in res['classes_acc']]

    # 论文目标值配置 (Table III for PU-PC, Table IV for HO13-HO18)
    if 'houston' in source_dataset.lower():
        paper_targets = {
            'C1': 84.88, 'C2': 40.41, 'C3': 51.65, 'C4': 97.27,
            'C5': 75.90, 'C6': 28.41,
            'Unk': 73.70, 'OS*': 63.09, 'HOS': 67.98
        }
        class_names = [
            "C1: Grass healthy", "C2: Grass stressed", "C3: Trees",
            "C4: Water", "C5: Residential buildings", "C6: Non-Residential buildings",
            "C7: Road (Unk)"
        ]
        hos_target = 67.98
        hos_threshold = 66.0
        os_star_threshold = 60.0
        unk_threshold = 70.0
    else:
        paper_targets = {
            'C1': 92.78, 'C2': 83.88, 'C3': 80.87, 'C4': 80.87,
            'C5': 84.35, 'C6': 98.54, 'C7': 77.82,
            'Unk': 87.54, 'OS*': 85.47, 'HOS': 86.49
        }
        class_names = [
            "C1: Tree", "C2: Asphalt", "C3: Brick", "C4: Bitumen",
            "C5: Shadow", "C6: Meadows", "C7: Bare Soil", "C8: Tiles (Unk)"
        ]
        hos_target = 86.49
        hos_threshold = 85.0
        os_star_threshold = 84.0
        unk_threshold = 85.0

    print("\n" + "=" * 70)
    print("                    UPOT 开放集域适应精度检测报告")
    print("=" * 70)
    print(f"{'指标名称':<20} | {'当前模型结果':<15} | {'论文UPOT目标':<15} | {'达成情况':<10}")
    print("-" * 70)
    print(f"{'HOS (调和均值)':<20} | {hos:>13.2f}% | {paper_targets['HOS']:>13.2f}% | {'[达标]' if hos >= hos_threshold else '[偏低]'}")
    print(f"{'OS* (已知类平均)':<20} | {os_star:>13.2f}% | {paper_targets['OS*']:>13.2f}% | {'[达标]' if os_star >= os_star_threshold else '[偏低]'}")
    print(f"{'Unk (未知类准确率)':<20} | {unk_acc:>13.2f}% | {paper_targets['Unk']:>13.2f}% | {'[达标]' if unk_acc >= unk_threshold else '[偏低]'}")
    print(f"{'OA (总体准确率)':<20} | {oa:>13.2f}% | {'-':>15} | -")
    print(f"{'AA (全类平均准确率)':<20} | {aa:>13.2f}% | {'-':>15} | -")
    print(f"{'Known OA (已知类总体)':<20} | {oa_known:>13.2f}% | {'-':>15} | -")
    print(f"{'Kappa 系数':<20} | {kappa:>13.2f}% | {'-':>15} | -")
    print("-" * 70)
    print("各类详细准确率:")
    for i, name in enumerate(class_names):
        key = f'C{i+1}' if i < len(source_known_classes) else 'Unk'
        target_str = f"{paper_targets[key]:.2f}%" if key in paper_targets else "-"
        acc_val = classes_acc[i] if i < len(classes_acc) else 0.0
        print(f"  {name:<30}: {acc_val:>6.2f}%  (论文目标: {target_str})")
    print("=" * 70)

    # 保存检测结果 JSON
    output_json = os.path.join(current_dir, f"eval_result_{source_dataset}-{target_dataset}_seed_{seed}.json")
    save_result = {
        'checkpoint': checkpoint_path,
        'seed': seed,
        'oa': oa,
        'aa': aa,
        'os_star': os_star,
        'unknown': unk_acc,
        'hos': hos,
        'kappa': kappa,
        'classes_acc': classes_acc,
        'paper_target_hos': paper_targets['HOS'],
        'eval_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    }
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(save_result, f, indent=4, ensure_ascii=False)
    print(f"\n检测结果已保存至: {output_json}")

    return save_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="UPOT Checkpoint Evaluation")
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/UPOT_PaviaU_7gt-PaviaC_OS_seed_58_best.pth',
                        help='模型权重文件路径 (.pth)')
    parser.add_argument('--batch_size', type=int, default=64, help='推理批大小')
    parser.add_argument('--device', type=int, default=0, help='GPU 设备编号')
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint, args.batch_size, args.device)
