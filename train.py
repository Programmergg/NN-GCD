import argparse
import copy
import math
import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler, AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root
from model import (
    DINOHead,
    info_nce_logits,
    SupConLoss,
    DistillLoss,
    ContrastiveLearningViewGenerator,
    get_params_groups,
    DRO_Loss,
    custom_regularization,
)



import torch
import os




from models import vision_transformer as vits

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train(backbone, projector, teacher, train_loader, test_loader, unlabelled_train_loader, args):
    student = nn.Sequential(backbone, projector).to(device)
    teacher = teacher.to(device)
    teacher.eval()
    params_groups = get_params_groups(student)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 1e-3,
    )
    ipe = len(train_loader)

    def ema_momentum_schedule(current_epoch, total_epochs, initial_momentum=0.7, final_momentum=0.999):
        momentum = final_momentum - (1 - initial_momentum) * (math.cos(math.pi * current_epoch / total_epochs) + 1) / 2
        return momentum

    momentum_scheduler = [ema_momentum_schedule(i, args.epochs * ipe) for i in range(args.epochs * ipe + 1)]
    momentum_scheduler = iter(momentum_scheduler)

    cluster_criterion = DistillLoss(
        args.warmup_teacher_temp_epochs,
        args.epochs,
        args.n_views,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.student_temp,
    )

    # Initialize best metrics
    best_test_acc_lab = 0
    best_train_acc_lab = 0
    best_train_acc_ubl = 0
    best_train_acc_all = 0

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        student.train()

        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_proj, student_out = student(images)
                teacher_enc = backbone(images)
                teacher_proj, teacher_out = teacher(teacher_enc)

                # Core logic for Loss calculation
                ADNCE_loss = DRO_Loss(
                    temperature=args.nce_temp,
                    tau_plus=0.1,
                    batch_size=args.batch_size,
                    beta=1.0,
                    estimator="adnce",
                    N=1000000,
                )

                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(args.n_views)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(args.n_views)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = -torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss

                contrastive_logits, contrastive_labels = info_nce_logits(
                    features=student_proj, n_views=args.n_views, temperature=args.nce_temp
                )
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                adnce_loss, _ = ADNCE_loss(student_proj)

                student_proj_div = torch.cat(
                    [f[mask_lab].unsqueeze(1) for f in student_proj.chunk(args.n_views)], dim=1
                )
                student_proj_norm = torch.nn.functional.normalize(student_proj_div, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_proj_norm, labels=sup_con_labels)

                # Regularization term
                reg_term = 0
                last_block = backbone.blocks[-1]
                for name, param in last_block.named_parameters():
                    if "weight" in name:
                        reg_term += custom_regularization(param, args.sparsity_weight, 1 - args.sparsity_weight)
                for name, param in projector.named_parameters():
                    if "weight" in name:
                        reg_term += custom_regularization(param, args.sparsity_weight, 1 - args.sparsity_weight)

                # NaN/Inf protection
                adnce_loss = torch.nan_to_num(adnce_loss, nan=0.0, posinf=0.0, neginf=0.0)
                cls_loss = torch.nan_to_num(cls_loss, nan=0.0, posinf=0.0, neginf=0.0)
                cluster_loss = torch.nan_to_num(cluster_loss, nan=0.0, posinf=0.0, neginf=0.0)
                sup_con_loss = torch.nan_to_num(sup_con_loss, nan=0.0, posinf=0.0, neginf=0.0)
                contrastive_loss = torch.nan_to_num(contrastive_loss, nan=0.0, posinf=0.0, neginf=0.0)
                reg_term = torch.nan_to_num(reg_term, nan=0.0, posinf=0.0, neginf=0.0)

                # Loss aggregation
                loss = 0
                loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                loss += (1 - args.sup_weight) * adnce_loss + args.sup_weight * sup_con_loss
                loss += reg_term * args.reg_weight

            # Backward propagation
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            # EMA update
            with torch.no_grad():
                m = next(momentum_scheduler)
                for param_q, param_k in zip(projector.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

            # Log printing
            if batch_idx % args.print_freq == 0:
                pstr = f"adnce_loss: {adnce_loss.item():.4f} cls_loss: {cls_loss.item():.4f} cluster_loss: {cluster_loss.item():.4f} sup_con_loss: {sup_con_loss.item():.4f} contrastive_loss: {contrastive_loss.item():.4f}"
                args.logger.info(
                    "Epoch: [{}][{}/{}]\t loss {:.5f}\t {}".format(
                        epoch, batch_idx, len(train_loader), loss.item(), pstr
                    )
                )

        # Epoch end logging
        args.logger.info("Train Epoch: {} Avg Loss: {:.4f} ".format(epoch, loss_record.avg))

        # Test unlabeled data
        args.logger.info("Testing on unlabelled examples in the training data...")
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            all_acc, old_acc, new_acc = test(
                backbone, teacher, unlabelled_train_loader, epoch=epoch, save_name="Train ACC Unlabelled", args=args
            )

        # Test labeled data
        args.logger.info("Testing on disjoint test set...")
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            all_acc_test, old_acc_test, new_acc_test = test(
                backbone, teacher, test_loader, epoch=epoch, save_name="Test ACC", args=args
            )

        # Print accuracy
        args.logger.info("Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}".format(all_acc, old_acc, new_acc))
        args.logger.info(
            "Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}".format(all_acc_test, old_acc_test, new_acc_test)
        )

        # Learning rate scheduling
        exp_lr_scheduler.step()

        # Save model
        teacher_model = nn.Sequential(backbone, teacher).to(device)
        save_dict = {
            "model": teacher_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
        }
        torch.save(save_dict, args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))

        # Save best model
        if all_acc > best_train_acc_all:
            args.logger.info(f"Best ACC on old Classes on disjoint test set: {old_acc_test:.4f}...")
            args.logger.info(
                "Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}".format(all_acc, old_acc, new_acc)
            )
            torch.save(save_dict, args.model_path[:-3] + f"_best.pt")
            args.logger.info("model saved to {}.".format(args.model_path[:-3] + f"_best.pt"))

            best_test_acc_lab = old_acc_test
            best_train_acc_lab = old_acc
            best_train_acc_ubl = new_acc
            best_train_acc_all = all_acc

        args.logger.info(f"Exp Name: {args.exp_name}")
        args.logger.info(
            f"Metrics with best model on test set: All: {best_train_acc_all:.4f} Old: {best_train_acc_lab:.4f} New: {best_train_acc_ubl:.4f}"
        )


def test(backbone, projector, test_loader, epoch, save_name, args):
    model = nn.Sequential(backbone, projector).to(device)
    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            features, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(
                mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label])
            )

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(
        y_true=targets, y_pred=preds, mask=mask, T=epoch, eval_funcs=args.eval_funcs, save_name=save_name, args=args
    )

    return all_acc, old_acc, new_acc


def clone_model(model):
    new_model = nn.Sequential(copy.deepcopy(model[0]), copy.deepcopy(model[1]))
    return new_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cluster", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=4, type=int)  # Reduce num_workers to decrease fork
    parser.add_argument("--eval_funcs", nargs="+", help="Which eval functions to use", default=["v2", "v2p"])

    parser.add_argument("--warmup_model_dir", type=str, default=None)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="scars",
        help="options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19",
    )
    parser.add_argument("--prop_train_labels", type=float, default=0.5)
    parser.add_argument("--use_ssb_splits", action="store_true", default=True)

    parser.add_argument("--grad_from_block", type=int, default=11)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--exp_root", type=str, default=exp_root)
    parser.add_argument("--transform", type=str, default="imagenet")
    parser.add_argument("--sup_weight", type=float, default=0.35)
    parser.add_argument("--reg_weight", type=float, default=3e-5)
    parser.add_argument("--sparsity_weight", type=float, default=0.6)
    parser.add_argument("--nce_temp", type=float, default=0.5)
    parser.add_argument("--n_views", default=2, type=int)

    parser.add_argument("--memax_weight", type=float, default=2)
    parser.add_argument("--warmup_teacher_temp", default=0.07, type=float)
    parser.add_argument("--teacher_temp", default=0.04, type=float)
    parser.add_argument("--warmup_teacher_temp_epochs", default=30, type=int)
    parser.add_argument("--student_temp", default=0.1, type=float)

    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--print_freq", default=10, type=int)
    parser.add_argument("--exp_name", default=None, type=str)

    # Initialize parameters
    args = parser.parse_args()
    device = torch.device("cuda:0")
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=["simgcd"])
    args.logger.info(f"Using evaluation function {args.eval_funcs[0]} to print results")

    torch.backends.cudnn.benchmark = True

    # Random seed
    args.interpolation = 3
    args.crop_pct = 0.875

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    setup_seed(42)

    # Proxy configuration (can be retained if needed)
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

    # Load DINO model
    backbone = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
    print(backbone)

    if args.warmup_model_dir is not None:
        args.logger.info(f"Loading weights from {args.warmup_model_dir}")
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location="cpu"))

    # Model parameter configuration
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes

    # Freeze model parameters (only fine-tune specified layers)
    for m in backbone.parameters():
        m.requires_grad = False
    for name, m in backbone.named_parameters():
        if "block" in name:
            block_num = int(name.split(".")[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    args.logger.info("model build")

    # Data augmentation
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(
        base_transform=train_transform, ori_transform=test_transform, n_views=args.n_views
    )

    # Load dataset
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(
        args.dataset_name, train_transform, test_transform, args
    )

    # Sampler
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=True,
        pin_memory=False,  # Disable pin_memory to avoid implicit fork
    )
    test_loader_unlabelled = DataLoader(
        unlabelled_train_examples_test, num_workers=args.num_workers, batch_size=256, shuffle=False, pin_memory=False
    )
    test_loader_labelled = DataLoader(
        test_dataset, num_workers=args.num_workers, batch_size=256, shuffle=False, pin_memory=False
    )

    # Projector head
    projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, use_bn=True, nlayers=args.num_mlp_layers)
    projector_target = DINOHead(
        in_dim=args.feat_dim, out_dim=args.mlp_out_dim, use_bn=True, nlayers=args.num_mlp_layers
    )

    # Start training
    train(backbone, projector, projector_target, train_loader, test_loader_labelled, test_loader_unlabelled, args)