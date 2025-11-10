# train_cifar10_transfer.py
# -*- coding: utf-8 -*-
"""
Transfer Learning on CIFAR-10 (Part 1 requirements)
- Dataset: CIFAR-10
- Pretrained models (choose one): resnet18 / vgg16 / densenet121 / mobilenet_v2 / efficientnet_b0
- Freeze vs. fine-tune switch
- Metrics: test accuracy, confusion matrix, precision/recall/F1
- ROC curves & AUC (OvR)
- Visualize misclassified samples
"""

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, transforms, models

# sklearn for metrics
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize

# ----------------------------
# Reproducibility (optional)
# ----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

print("Torch:", torch.__version__)
print("CUDA runtime in torch:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())

cudnn.benchmark = True  # speed up on fixed input size

# ----------------------------
# ### >>> [MOD-CONFIG] 可調參數（模型/凍結策略/訓練回合/學習率等）
# ----------------------------
# 可改為你想要的模型名稱：'resnet18' / 'vgg16' / 'densenet121' / 'mobilenet_v2' / 'efficientnet_b0'
MODEL_NAME = 'vgg16'      # <<< 你可改
FREEZE_BACKBONE = True       # <<< True=只訓練分類頭；False=全模型微調
BATCH_SIZE = 128
EPOCHS = 2                  # <<< 你可改
BASE_LR = 1e-3               # <<< 你可改
STEP_SIZE = 7
GAMMA = 0.1
NUM_WORKERS = 0              # Windows 建議 0
DATA_ROOT = "./data"         # 下載/存放 CIFAR-10 的目錄
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# CIFAR-10 類別
CIFAR10_CLASSES = ('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')
NUM_CLASSES = len(CIFAR10_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# ----------------------------
# ### >>> [MOD-DATASET] 將原本 ImageFolder(螞蟻/蜜蜂) 改為 CIFAR-10
# - 也把 32x32 resize 到 224 以符合 ImageNet 預訓練模型輸入大小
# ----------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),       # <<< CIFAR-10 常見增強
    transforms.RandomHorizontalFlip(),          # <<< CIFAR-10 常見增強
    transforms.Resize(224),                     # <<< 變 224x224
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

test_tf = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# 下載/載入 CIFAR-10
train_full = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=train_tf)
test_set   = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=test_tf)

# 建立 validation split (例如 45k train / 5k val)
val_ratio = 5000                                # <<< 你可改
train_len = len(train_full) - val_ratio
train_set, val_set = torch.utils.data.random_split(
    train_full, [train_len, val_ratio], generator=torch.Generator().manual_seed(SEED)
)

def make_loader(ds, shuffle):
    return torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

train_loader = make_loader(train_set, True)
val_loader   = make_loader(val_set, False)
test_loader  = make_loader(test_set, False)

dataset_sizes = {
    'train': len(train_set),
    'val': len(val_set),
    'test': len(test_set)
}

# ----------------------------
# ### >>> [MOD-MODEL] 模型工廠＋替換最後分類層為 10 類
# ----------------------------
def build_model(name: str, num_classes: int, freeze_backbone: bool = True):
    name = name.lower()
    if name == 'resnet18':
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # <<< 預訓練權重
        in_feats = net.fc.in_features
        net.fc = nn.Linear(in_feats, num_classes)                              # <<< 改為 10 類
        backbone_params = [p for n,p in net.named_parameters() if not n.startswith('fc')]
    elif name == 'vgg16':
        net = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        in_feats = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_feats, num_classes)                  # <<< 改為 10 類
        backbone_params = list(net.features.parameters())
    elif name == 'densenet121':
        net = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_feats = net.classifier.in_features
        net.classifier = nn.Linear(in_feats, num_classes)                      # <<< 改為 10 類
        backbone_params = [p for n,p in net.named_parameters() if not n.startswith('classifier')]
    elif name == 'mobilenet_v2':
        net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        in_feats = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_feats, num_classes)                  # <<< 改為 10 類
        backbone_params = [p for n,p in net.named_parameters() if not n.startswith('classifier')]
    elif name == 'efficientnet_b0':
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_feats = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_feats, num_classes)                  # <<< 改為 10 類
        backbone_params = [p for n,p in net.named_parameters() if not n.startswith('classifier')]
    else:
        raise ValueError(f"Unknown model name: {name}")

    # ### >>> [MOD-FREEZE] 控制是否凍結骨幹
    if freeze_backbone:
        for p in backbone_params:
            p.requires_grad = False

    return net

model = build_model(MODEL_NAME, NUM_CLASSES, freeze_backbone=FREEZE_BACKBONE).to(device)

# ### >>> [MOD-OPT] 只優化 requires_grad=True 的參數（凍結策略生效）
params_to_update = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params_to_update, lr=BASE_LR, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
criterion = nn.CrossEntropyLoss()

best_ckpt_path = os.path.join(SAVE_DIR, f"best_{MODEL_NAME}_{'frozen' if FREEZE_BACKBONE else 'finetune'}.pt")

# ----------------------------
# Utilities
# ----------------------------
def imshow_tensor(img_tensor, title=None):
    """Denormalize and show tensor image (CHW)."""
    img = img_tensor.clone().detach().cpu().numpy().transpose(1,2,0)
    img = (img * np.array(IMAGENET_STD)) + np.array(IMAGENET_MEAN)
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title: plt.title(title)
    plt.axis('off')

# ----------------------------
# ### >>> [MOD-TRAIN] 輕量化訓練/驗證函式（支援 scheduler、凍結策略）
# ----------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss, running_correct = 0.0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = outputs.max(1)
        running_loss += loss.item() * inputs.size(0)
        running_correct += (preds == labels).sum().item()

    loss = running_loss / len(loader.dataset)
    acc  = running_correct / len(loader.dataset)
    return loss, acc

@torch.no_grad()
def evaluate(model, loader, criterion):
    # ### >>> [MOD-EVAL] 回傳 y_true/y_pred/y_prob 以便後續計算多種指標
    model.eval()
    running_loss, running_correct = 0.0, 0
    all_labels, all_preds, all_probs = [], [], []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)

        running_loss += loss.item() * inputs.size(0)
        running_correct += (preds == labels).sum().item()

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    loss = running_loss / len(loader.dataset)
    acc  = running_correct / len(loader.dataset)
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)  # shape [N, num_classes]
    return loss, acc, y_true, y_pred, y_prob

# ----------------------------
# ### >>> [MOD-METRICS] 典型指標繪圖（混淆矩陣、ROC/AUC、誤分類樣本）
# ----------------------------
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(7.5,6))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label'); plt.xlabel('Predicted label')
    plt.tight_layout()

def plot_roc_curves(y_true, y_prob, classes, title='ROC Curves (OvR)'):
    # Binarize labels for OvR ROC
    y_bin = label_binarize(y_true, classes=list(range(len(classes))))
    # 某些類別可能沒有樣本，安全檢查
    valid_cols = [i for i in range(y_prob.shape[1]) if y_bin[:, i].sum() > 0]
    plt.figure(figsize=(7.5,6))
    for i in valid_cols:
        from sklearn.metrics import roc_curve, roc_auc_score  # 再引一次避免 IDE 提示
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc_i = roc_auc_score(y_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f"{classes[i]} (AUC={auc_i:.3f})")
    # micro-average
    try:
        from sklearn.metrics import roc_auc_score
        auc_micro = roc_auc_score(y_bin[:, valid_cols], y_prob[:, valid_cols], average='micro')
        plt.plot([0,1],[0,1],'--')
        plt.title(f"{title} | micro-AUC={auc_micro:.3f}")
    except Exception:
        plt.plot([0,1],[0,1],'--'); plt.title(title)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right'); plt.tight_layout()

def show_misclassified(model, loader, classes, max_images=16):
    model.eval()
    imgs, preds_txt, trues_txt = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs); _, preds = outputs.max(1)
            mismatch = preds != labels
            if mismatch.any():
                mis_idx = mismatch.nonzero(as_tuple=False).squeeze(1)
                for idx in mis_idx:
                    imgs.append(inputs[idx].cpu())
                    preds_txt.append(classes[preds[idx].item()])
                    trues_txt.append(classes[labels[idx].item()])
                    if len(imgs) >= max_images: break
            if len(imgs) >= max_images: break

    if not imgs:
        print("No misclassified samples found (great!)")
        return
    cols = 4
    rows = int(np.ceil(len(imgs)/cols))
    plt.figure(figsize=(cols*3, rows*3))
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        imshow_tensor(img, title=f"pred:{preds_txt[i]}\ntrue:{trues_txt[i]}")
    plt.tight_layout()

# ----------------------------
# ### >>> [MOD-LOOP] 簡潔訓練主迴圈 + 最佳模型儲存/載入
# ----------------------------
best_acc = 0.0
for epoch in range(EPOCHS):
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion)
    scheduler.step()

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), best_ckpt_path)

    dt = time.time() - t0
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
          f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} | "
          f"{dt/60:.1f}m")

print(f"Best Val Acc: {best_acc:.4f}, checkpoint saved to: {best_ckpt_path}")

# ----------------------------
# ### >>> [MOD-TEST] 測試集評估 + 報表 + 圖表輸出
# ----------------------------
# load best
model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
test_loss, test_acc, y_true, y_pred, y_prob = evaluate(model, test_loader, criterion)
print(f"[TEST] Loss {test_loss:.4f} Acc {test_acc:.4f}")

# Classification report (per-class precision/recall/F1)
report = classification_report(y_true, y_pred, target_names=CIFAR10_CLASSES, digits=4)
print("\nClassification Report:\n", report)

# Compute per-class precision / recall / f1 / support
precisions, recalls, f1s, supports = precision_recall_fscore_support(
    y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
)

# Print a neat table
line_fmt = "{:<12} {:>9} {:>9} {:>9} {:>8}"
print("\nPer-class metrics:")
print(line_fmt.format('class', 'precision', 'recall', 'f1-score', 'support'))
print('-'*52)
for cls_name, p, r, f1s_val, s in zip(CIFAR10_CLASSES, precisions, recalls, f1s, supports):
    print(line_fmt.format(cls_name, f"{p:.4f}", f"{r:.4f}", f"{f1s_val:.4f}", int(s)))

# Plot grouped bar chart for precision / recall / f1 (with numeric labels)
indices = np.arange(NUM_CLASSES)
width = 0.25
plt.figure(figsize=(12, 6))
bars_p = plt.bar(indices - width, precisions, width, label='precision')
bars_r = plt.bar(indices, recalls, width, label='recall')
bars_f = plt.bar(indices + width, f1s, width, label='f1-score')
plt.xticks(indices, CIFAR10_CLASSES, rotation=45, ha='right')
plt.ylim(0.0, 1.05)
plt.ylabel('score')
plt.title(f'Per-class precision/recall/f1 - {MODEL_NAME}')
plt.legend(loc='upper right')

# annotate bars with values
def annotate_bars(bars, fmt="{:.2f}", offset=0.01):
    for bar in bars:
        h = bar.get_height()
        if np.isnan(h):
            txt = "n/a"
        else:
            txt = fmt.format(h)
        plt.text(bar.get_x() + bar.get_width()/2., h + offset, txt,
                 ha='center', va='bottom', fontsize=8, rotation=0)

annotate_bars(bars_p, fmt="{:.2f}", offset=0.01)
annotate_bars(bars_r, fmt="{:.2f}", offset=0.01)
annotate_bars(bars_f, fmt="{:.2f}", offset=0.01)

plt.tight_layout()
chart_path = os.path.join(SAVE_DIR, f"per_class_metrics_{MODEL_NAME}.png")
plt.savefig(chart_path, dpi=200)
plt.close()
print(f"\nPer-class metric chart saved to: {chart_path}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, CIFAR10_CLASSES, normalize=False, title="Confusion Matrix (counts)")
plt.savefig(os.path.join(SAVE_DIR, f"cm_counts_{MODEL_NAME}.png"), dpi=200)
plt.close()

plot_confusion_matrix(cm, CIFAR10_CLASSES, normalize=True, title="Confusion Matrix (normalized)")
plt.savefig(os.path.join(SAVE_DIR, f"cm_norm_{MODEL_NAME}.png"), dpi=200)
plt.close()

# ROC curves & AUC (OvR)
plot_roc_curves(y_true, y_prob, CIFAR10_CLASSES, title=f"ROC Curves (OvR) - {MODEL_NAME}")
plt.savefig(os.path.join(SAVE_DIR, f"roc_{MODEL_NAME}.png"), dpi=200)
plt.close()

# Visualize misclassified samples
show_misclassified(model, test_loader, CIFAR10_CLASSES, max_images=16)
plt.savefig(os.path.join(SAVE_DIR, f"misclassified_{MODEL_NAME}.png"), dpi=200)
plt.close()

print("All figures saved to:", SAVE_DIR)
