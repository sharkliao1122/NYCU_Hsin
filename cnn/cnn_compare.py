#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cnn_compare.py
---------------------------------
Train and compare VGG16 / ResNet18 / MobileNetV2 on the same dataset,
under identical training budget, then report the best model.

Dataset layout (ImageFolder):
data_root/
  train/
    class_a/ *.jpg
    class_b/ *.jpg
    ...
  val/
    class_a/ *.jpg
    class_b/ *.jpg
  test/
    class_a/ *.jpg
    class_b/ *.jpg

Example:
python cnn_compare.py --data_root data_root --epochs 25 --lr 1e-3 --batch_size 64

Author: ChatGPT (cnn project helper)
"""
import os
import math
import time
import json
import argparse
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Optional dependency: scikit-learn for micro-AUC
try:
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_model(name: str, num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    """Create a classification model by name."""
    name = name.lower()
    if name == 'vgg16':
        net = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for p in net.features.parameters():
                p.requires_grad = False
        in_f = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_f, num_classes)
        return net

    if name == 'resnet18':
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for p in net.layer1.parameters(): p.requires_grad = False
            for p in net.layer2.parameters(): p.requires_grad = False
            for p in net.layer3.parameters(): p.requires_grad = False
            for p in net.layer4.parameters(): p.requires_grad = False
            for p in net.conv1.parameters():  p.requires_grad = False
            for p in net.bn1.parameters():    p.requires_grad = False
        in_f = net.fc.in_features
        net.fc = nn.Linear(in_f, num_classes)
        return net

    if name == 'mobilenet_v2':
        net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for p in net.features.parameters():
                p.requires_grad = False
        in_f = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_f, num_classes)
        return net

    raise ValueError(f"Unknown model name: {name}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    y_prob_all: List[List[float]] = []

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += inputs.size(0)

        y_true_all.extend(labels.detach().cpu().tolist())
        y_pred_all.extend(preds.detach().cpu().tolist())
        y_prob_all.extend(torch.softmax(outputs, dim=1).detach().cpu().tolist())

    epoch_loss = running_loss / max(1, total)
    epoch_acc = running_corrects / max(1, total)
    return epoch_loss, epoch_acc, y_true_all, y_pred_all, y_prob_all


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device: torch.device, scaler=None):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += inputs.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = running_corrects / max(1, total)
    return epoch_loss, epoch_acc


def make_loaders(data_root: str, img_size: int, batch_size: int, num_workers: int = 4):
    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tfm = transforms.Compose([
        transforms.Resize(int(img_size*1.14)),
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tfm = transforms.Compose([
        transforms.Resize(int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_root, 'train'), transform=train_tfm)
    val_ds   = datasets.ImageFolder(os.path.join(data_root, 'val'),   transform=test_tfm)
    test_ds  = datasets.ImageFolder(os.path.join(data_root, 'test'),  transform=test_tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.classes


def compute_micro_auc(y_true: List[int], y_prob: List[List[float]], num_classes: int):
    if not HAS_SKLEARN:
        return float('nan')
    try:
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        auc = roc_auc_score(y_bin, y_prob, average='micro', multi_class='ovr')
        return float(auc)
    except Exception:
        return float('nan')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='root with train/val/test folders')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--models', type=str, default='vgg16,resnet18,mobilenet_v2',
                        help='comma-separated: vgg16,resnet18,mobilenet_v2')
    parser.add_argument('--step_size', type=int, default=10, help='StepLR step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='StepLR gamma')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--out_dir', type=str, default='runs')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Data
    train_loader, val_loader, test_loader, class_names = make_loaders(
        args.data_root, args.img_size, args.batch_size, num_workers=min(8, os.cpu_count() or 4)
    )
    num_classes = len(class_names)
    print(f'Classes: {class_names} (num={num_classes})')

    # Models to try
    model_names = [m.strip() for m in args.models.split(',') if m.strip()]
    results = []  # (name, best_val, test_acc, micro_auc, best_ckpt)

    for name in model_names:
        print('\n' + '='*72)
        print(f'Training: {name} | epochs={args.epochs} | lr={args.lr} | freeze={args.freeze_backbone}')
        print('='*72)

        model = build_model(name, num_classes, freeze_backbone=args.freeze_backbone).to(device)
        params_to_update = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(params_to_update, lr=args.lr, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler() if (args.use_amp and device.type == 'cuda') else None

        best_val = 0.0
        best_path = os.path.join(args.out_dir, f'best_{name}_{"frozen" if args.freeze_backbone else "finetune"}.pt')

        for epoch in range(1, args.epochs+1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
            val_loss, val_acc, *_ = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(), best_path)

            dt = time.time() - t0
            print(f'Epoch {epoch:02d}/{args.epochs} | '
                  f'Train Acc {tr_acc:.4f} | Val Acc {val_acc:.4f} (best {best_val:.4f}) | {dt:.1f}s')

        # Test with best checkpoint
        model.load_state_dict(torch.load(best_path, map_location=device))
        test_loss, test_acc, y_true, y_pred, y_prob = evaluate(model, test_loader, criterion, device)
        micro_auc = compute_micro_auc(y_true, y_prob, num_classes)
        print(f'[{name}] TEST Acc: {test_acc:.4f} | micro-AUC: {micro_auc:.4f}')

        results.append((name, best_val, test_acc, micro_auc, best_path))

        # free memory for next model
        if device.type == 'cuda':
            del model
            torch.cuda.empty_cache()

    # Summary
    print('\n=== Summary (higher is better) ===')
    header = f'{"model":<14}{"best_val":>12}{"test_acc":>12}{"micro_AUC":>12}'
    print(header)
    for (n, bva, ta, auc, _) in results:
        print(f'{n:<14}{bva:>12.4f}{ta:>12.4f}{auc:>12.4f}')

    best_by_val = max(results, key=lambda x: x[1])
    print(f'\n>> Best by Val Acc: {best_by_val[0]} | Val={best_by_val[1]:.4f} '
          f'| Test={best_by_val[2]:.4f} | micro-AUC={best_by_val[3]:.4f}')

    # Save JSON report
    report = {
        "classes": class_names,
        "results": [
            {"model": n, "best_val": float(bva), "test_acc": float(ta), "micro_auc": float(auc), "ckpt": ckpt}
            for (n, bva, ta, auc, ckpt) in results
        ],
        "best_model": {
            "by_val_acc": {
                "model": best_by_val[0],
                "best_val": float(best_by_val[1]),
                "test_acc": float(best_by_val[2]),
                "micro_auc": float(best_by_val[3]),
                "ckpt": best_by_val[4]
            }
        }
    }
    json_path = os.path.join(args.out_dir, 'summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'\nSaved summary to: {json_path}')
    print('Done.')


if __name__ == '__main__':
    main()
