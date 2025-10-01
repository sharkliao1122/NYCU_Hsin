# ReLU 激活 + Dropout
import torch  # 匯入 PyTorch 主程式庫
import torchvision  # 匯入 torchvision 影像資料集與工具
import torchvision.transforms as transforms  # 匯入影像轉換工具
import matplotlib.pyplot as plt  # 匯入繪圖工具
import numpy as np  # 匯入數值運算工具
import torch.nn as nn  # 匯入神經網路模組
import torch.nn.functional as F  # 匯入常用函式
import torch.optim as optim  # 匯入最佳化工具
from collections import Counter  # 匯入計數器工具
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report

# 定義影像前處理流程：轉為 Tensor 並標準化到 [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 4  # 每批次取 4 張影像

 # 載入 CIFAR-10 訓練集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)


# num_workers=0 避免 Windows 系統的多工問題，非0可以加速資料載入

# 載入 CIFAR-10 測試集
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

 # CIFAR-10 的 10 個類別
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 統計每個類別的圖片數量
label_counter = Counter(trainset.targets)
for idx, class_name in enumerate(classes):
    print(f'{class_name:5s}: {label_counter[idx]}')


# 顯示影像的輔助函式
def imshow(img):
    img = img / 2 + 0.5     # 反標準化，將像素值還原到 [0,1]
    npimg = img.numpy()     # 轉換為 numpy 陣列
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 調整通道順序並顯示 從 C,H,W 轉為 H,W,C (H 是高度 W 是寬度 C 是通道數)
    plt.show()

# 取得一批隨機訓練影像並顯示
dataiter = iter(trainloader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))# 顯示標籤(類別)

# 定義卷積神經網路
class NetReLU(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 2)
        self.conv4 = nn.Conv2d(32, 64, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(p=dropout_p)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

net = NetReLU(dropout_p=0.3)  # 30% dropout

# 優化器實驗：SGD、Adam、RMSprop
optimizers = {
    'SGD': lambda net: optim.SGD(net.parameters(), lr=0.001, momentum=0.9),
    'Adam': lambda net: optim.Adam(net.parameters(), lr=0.001),
    'RMSprop': lambda net: optim.RMSprop(net.parameters(), lr=0.001)
}

num_epochs = 5  # 可自行調整訓練 epoch 數
train_loss_history = {}
train_acc_history = {}

for opt_name, opt_fn in optimizers.items():
    print(f'\n===== Optimizer: {opt_name} =====')
    # 重新初始化模型
    net = NetReLU(dropout_p=0.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = opt_fn(net)
    loss_list = []
    acc_list = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        avg_loss = running_loss / len(trainloader)
        avg_acc = 100 * correct / total
        loss_list.append(avg_loss)
        acc_list.append(avg_acc)
        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={avg_acc:.2f}%')
    train_loss_history[opt_name] = loss_list
    train_acc_history[opt_name] = acc_list
print('Finished Training All Optimizers')

# 繪製 loss 與 accuracy 曲線
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for opt_name, loss_list in train_loss_history.items():
    plt.plot(loss_list, label=opt_name)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
for opt_name, acc_list in train_acc_history.items():
    plt.plot(acc_list, label=opt_name)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.show()

# 儲存訓練好的模型
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# 取得一批測試影像並顯示
dataiter = iter(testloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

# 載入模型並進行預測
net = NetReLU(dropout_p=0.5)  # 50% dropout
net.load_state_dict(torch.load(PATH))
outputs = net(images)  # 前向傳播
_, predicted = torch.max(outputs, 1)  # 取得預測類別
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

# 計算整體測試集準確率
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


# 混淆矩陣、精確率、召回率、F1-score、誤分類可視化


all_preds = []
all_labels = []
misclassified_imgs = []
misclassified_true = []
misclassified_pred = []

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        # 收集誤分類圖片
        for img, true, pred in zip(images, labels, predictions):
            if true != pred:
                misclassified_imgs.append(img)
                misclassified_true.append(true.item())
                misclassified_pred.append(pred.item())

# 混淆矩陣
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 精確率、召回率、F1-score
report = classification_report(all_labels, all_preds, target_names=classes, digits=3)
print(report)

# 顯示部分誤分類圖片
num_show = 8  # 可調整展示數量
plt.figure(figsize=(16,4))
for i in range(min(num_show, len(misclassified_imgs))):
    plt.subplot(1, num_show, i+1)
    img = misclassified_imgs[i] / 2 + 0.5  # 反標準化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.title(f'T:{classes[misclassified_true[i]]}\nP:{classes[misclassified_pred[i]]}')
    plt.axis('off')
plt.suptitle('Misclassified Images (T=True, P=Pred)')
plt.show()

# 設定運算裝置（GPU 或 CPU）
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)  # 將模型移到 GPU
inputs, labels = inputs.to(device), labels.to(device)  # 將資料移到 GPU
print(device)


del dataiter  # 釋放記憶體
