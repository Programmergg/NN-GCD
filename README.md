<div align="center">

# A Fresh Look at Generalized Category Discovery through Non-negative Matrix Factorization

[![arXiv](https://img.shields.io/badge/arXiv-2410.21807-b31b1b.svg)](https://arxiv.org/abs/2410.21807)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#installation)

Zhong Ji, Shuo Yang, Jingren Liu, Yanwei Pang, Jungong Han

</div>

---

## 🔥 News
- **[2026-03-09]** Repository initialized.
- **[2026-03-09]** README and project structure released.
- **[2026-03-11]** Training and evaluation code will be released.
- **[2026-03-11]** Reproducibility guide will be released.

---

## Overview

Generalized Category Discovery (GCD) aims to recognize both **seen (base)** and **unseen (novel)** categories by leveraging labeled data from base classes together with unlabeled data containing both old and new classes. This repository implements **NN-GCD**, a simple yet effective framework that revisits GCD from a **non-negative matrix factorization (NMF)** perspective. The method connects:

- **Optimal K-means**
- **Symmetric Non-negative Matrix Factorization (SNMF)**
- **Non-negative Contrastive Learning**

and introduces tailored optimization objectives for generalized category discovery.

---

## Features

- Training code for GCD benchmarks
- Single-GPU training pipeline
- Multi-GPU distributed training pipeline
- Support for standard GCD datasets
- Backbone + projector based NN-GCD implementation
- Evaluation on both old and new classes

---

## Repository Structure

The current repository structure is organized as follows:

```text
NNGCD/
├── data/                   # dataset storage / augmentation / dataset utilities
├── dataloader/             # dataset and dataloader implementations
├── models/                 # backbone definitions and model-related modules
├── scripts/                # helper scripts for launching experiments
├── util/                   # logging, evaluation, samplers, utilities
├── config.py               # global config (e.g. experiment root)
├── model.py                # core loss functions / heads / view generators / helpers
├── train.py                # single-GPU training entry
├── train_mp.py             # multi-GPU distributed training entry
└── requirements.txt         # dependencies
```

---

## Installation

### 1. Create environment

```bash
conda create -n nngcd python=3.10 -y
conda activate nngcd
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Datasets

This repository supports common GCD benchmarks, including: CIFAR-10, CIFAR-100, ImageNet-100, CUB, Stanford Cars, FGVC Aircraft and Herbarium-19. Please download the datasets and place them under the `data/` directory.

A recommended structure is:

```text
data/
├── cifar/
├── cub/
├── fgvc_aircraft/
├── herbarium_19/
├── imagenet/
├── stanford_cars/
└── ssb_splits/
```

---

## Training

This project provides **two training modes**:

1. **Single-GPU / standard training** using `train.py`
2. **Multi-GPU distributed training** using `train_mp.py`

Use `train.py` when you want to run experiments on a single GPU.

## Example

```bash
python train.py \
  --dataset_name cifar10 \
  --batch_size 128 \
  --epochs 200 \
  --lr 0.1 \
  --momentum 0.9 \
  --weight_decay 1e-4 \
  --transform imagenet \
  --sup_weight 0.35 \
  --reg_weight 3e-5 \
  --sparsity_weight 0.6 \
  --nce_temp 0.5 \
  --n_views 2 \
  --memax_weight 2 \
  --warmup_teacher_temp 0.07 \
  --teacher_temp 0.04 \
  --warmup_teacher_temp_epochs 30 \
  --student_temp 0.1 \
  --fp16 \
  --print_freq 10 \
  --exp_name nngcd_cifar10_single
```

Use `train_mp.py` for **distributed training** across multiple GPUs.

## Example (4 GPUs on one machine)

```bash
torchrun --nproc_per_node=4 train_mp.py \
  --dataset_name cifar10 \
  --batch_size 128 \
  --epochs 200 \
  --lr 0.1 \
  --momentum 0.9 \
  --weight_decay 1e-4 \
  --transform imagenet \
  --sup_weight 0.35 \
  --n_views 2 \
  --memax_weight 2 \
  --warmup_teacher_temp 0.07 \
  --teacher_temp 0.04 \
  --warmup_teacher_temp_epochs 30 \
  --fp16 \
  --print_freq 10 \
  --exp_name nngcd_cifar10_ddp
```

A pretrained or previously saved checkpoint can be loaded using `--warmup_model_dir`.

## Example

```bash
python train.py \
  --dataset_name cifar10 \
  --warmup_model_dir path/to/checkpoint.pt \
  --exp_name eval_or_finetune_run
```

or for distributed mode:

```bash
torchrun --nproc_per_node=4 train_mp.py \
  --dataset_name cifar10 \
  --warmup_model_dir path/to/checkpoint.pt \
  --exp_name eval_or_finetune_run_ddp
```

---

## Citation

If you find this repository useful, please cite:

```bibtex
@article{ji2026fresh,
  title={A fresh look at generalized category discovery through non-negative matrix factorization},
  author={Ji, Zhong and Yang, Shuo and Liu, Jingren and Pang, Yanwei and Tang, Chen},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2026},
  publisher={IEEE}
}
```


---

## ⭐ Star History

<p align="center">
  <a href="https://star-history.com/Programmergg/NTK-CL&Date">
    <img src="https://api.star-history.com/svg?repos=Programmergg/NTK-CL&type=Date" width="650" alt="Star History Chart">
  </a>
</p>

---

## 📬 Contact

For questions and discussions:
- Email: **jrl0219@tju.edu.cn**
- Issues: please open a GitHub Issue in this repo.

---
