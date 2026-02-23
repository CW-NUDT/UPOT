# Uncertainty-Aware Partial Optimal Transport for Open-Set Cross-Domain Hyperspectral Image Classification

> **News 📢:** This repository contains the official open-source implementation for the paper *"Uncertainty-Aware Partial Optimal Transport for Open-Set Cross-Domain Hyperspectral Image Classification"*. 
> 
> **⚠️ Note: The full source code will be released and made publicly available in this repository upon the formal publication of the paper.**

## Requirements

- Python 3.7+
- PyTorch 1.0+
- torchmetrics 0.10.3
- numpy
- scikit-learn
- OT 
- tqdm

## Installation

Clone the repository and install the dependencies:

## Datasets

The method supports the following hyperspectral datasets:

- **Pavia University **: 103 bands, 9 classes
- **Pavia Center **: 102 bands, 9 classes  
- **Houston 2013**: 48 bands, 7 classes
- **Houston 2018**: 48 bands, 7 classes

Please download and place the dataset files in the `datasets/` directory before running the code.

## Usage

### Training

You can start the training process by running the following command:

```bash
python UPOT_train.py --source_dataset PaviaU_7gt --target_dataset PaviaC_OS
```
