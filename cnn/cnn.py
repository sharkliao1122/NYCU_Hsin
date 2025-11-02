# License: BSD
# Author: Sasank Chilamkurthy

# License: BSD
# Author: Sasank Chilamkurthy

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
from tempfile import TemporaryDirectory

cudnn.benchmark = True
plt.ion()   # interactive mode

# ---------- CONFIG ----------
# Path to dataset (keep as your absolute path)
data_dir = "C:\Users\s7103\OneDrive\桌面\碩士班\NYCU_Hsin\cnn\computer vision\hymenoptera_data"

# Outputs directory under the cnn module folder
output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(output_dir, exist_ok=True)

print(f"Outputs will be saved to: {output_dir}")

# ---------- DATA LOAD ----------
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Check dataset path early and give clear error if missing
if not os.path.isdir(data_dir):
    raise FileNotFoundError(f"data_dir not found: {data_dir}")

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

# 在 Windows/互動環境下 num_workers>0 有時會造成問題，預設改為 0（必要時再調整）
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=0)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# Use available accelerator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# ---------- DATA VISUALIZATION ----------
def imshow(inp, title=None):
    """Display image for Tensor or numpy array."""
    if isinstance(inp, torch.Tensor):
        inp = inp.detach().cpu().numpy()
    inp = inp.transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        # title 可能是 list（batch 用），把 list 轉成字串
        plt.title(title if isinstance(title, str) else " | ".join(map(str, title)))
    plt.pause(0.001)  # pause a bit so that plots are updated

# Sanity check: display one batch (will error if dataloader empty)
try:
    inputs, classes = next(iter(dataloaders['train']))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x.item()] for x in classes])
except Exception as e:
    print(f"Skipping batch visual check: {e}")

# ---------- TRAINING FUNCTION ----------
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # persistent checkpoint file under outputs
    best_model_params_path = os.path.join(output_dir, 'best_model_params.pt')
    # write an initial checkpoint
    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = torch.tensor(0, dtype=torch.long, device=device)

            # Iterate over data.
            for inputs_batch, labels in dataloaders[phase]:
                inputs_batch = inputs_batch.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs_batch)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs_batch.size(0)
                running_corrects += torch.sum(preds == labels.data).to(device)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / max(1, dataset_sizes[phase])
            # convert running_corrects to scalar
            epoch_acc = running_corrects.double().item() / max(1, dataset_sizes[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)
                print(f"Saved improved model to: {best_model_params_path}")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights（安全對應 CPU/GPU）
    model.load_state_dict(torch.load(best_model_params_path, map_location=device))
    print(f"Loaded best model from: {best_model_params_path}")
    return model

# ---------- VISUALIZATION UTIL ----------
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(8, 8))

    with torch.no_grad():
        for i, (inputs_batch, labels) in enumerate(dataloaders['val']):
            inputs_batch = inputs_batch.to(device)
            labels = labels.to(device)

            outputs = model(inputs_batch)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs_batch.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j].item()]}')
                imshow(inputs_batch.cpu().detach()[j])

                if images_so_far == num_images:
                    # save visualization to outputs folder
                    fname = os.path.join(output_dir, f'val_predictions_{int(time.time())}.png')
                    fig.tight_layout()
                    fig.savefig(fname, dpi=150)
                    print(f"Saved validation prediction grid to: {fname}")
                    model.train(mode=was_training)
                    plt.close(fig)
                    return
        model.train(mode=was_training)
    plt.close(fig)

# ---------- MODEL SETUP ----------
# Robust weight loading: prefer new enum API, fallback to pretrained for older torchvision
try:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model_ft = models.resnet18(weights=weights)
except Exception:
    model_ft = models.resnet18(pretrained=True)

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

try:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model_conv = models.resnet18(weights=weights)
except Exception:
    model_conv = models.resnet18(pretrained=True)

for param in model_conv.parameters():
    param.requires_grad = False

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)
model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

# ---------- TRAIN AND EVALUATE ----------
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)

visualize_model(model_conv)

plt.ioff()
plt.show()

# ---------- SINGLE-IMAGE PREDICTION ----------
def visualize_model_predictions(model, img_path):
    was_training = model.training
    model.eval()

    if not os.path.isfile(img_path):
        print(f"Image not found: {img_path}")
        model.train(mode=was_training)
        return

    img = Image.open(img_path).convert('RGB')
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        fig = plt.figure(figsize=(4, 4))
        ax = plt.subplot(1, 1, 1)
        ax.axis('off')
        ax.set_title(f'Predicted: {class_names[preds[0].item()]}')
        imshow(img.cpu().detach()[0])

        # save the single-image prediction
        imgname = os.path.splitext(os.path.basename(img_path))[0]
        outpath = os.path.join(output_dir, f'pred_{imgname}_{int(time.time())}.png')
        plt.gcf().tight_layout()
        plt.savefig(outpath, dpi=150)
        print(f"Saved single-image prediction to: {outpath}")
        plt.close(fig)

        model.train(mode=was_training)

# Call single-image demo if sample image exists inside dataset
sample_img = os.path.join(data_dir, 'val', 'bees', '72100438_73de9f17af.jpg')
if os.path.isfile(sample_img):
    visualize_model_predictions(model_conv, img_path=sample_img)
else:
    print(f"Sample image not found, skipping single-image demo: {sample_img}")