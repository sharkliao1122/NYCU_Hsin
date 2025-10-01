# -*- coding: utf-8 -*-

# For tips on running notebooks in Google Colab, see

# -*- coding: utf-8 -*-

# 本程式示範如何用 PyTorch 訓練 CIFAR-10 影像分類器
# 參考官方教學：https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch  # 匯入 PyTorch 主程式庫
import torchvision  # 匯入 torchvision 影像資料集與工具
import torchvision.transforms as transforms  # 匯入影像轉換工具
import matplotlib.pyplot as plt  # 匯入繪圖工具
import numpy as np  # 匯入數值運算工具
import torch.nn as nn  # 匯入神經網路模組
import torch.nn.functional as F  # 匯入常用函式
import torch.optim as optim  # 匯入最佳化工具
# 顯示 CUDA 相關資訊

print(torch.version.cuda)         # 顯示 PyTorch 支援的 CUDA 版本
print(torch.cuda.is_available())  # True 表示可用


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


# 定義卷積神經網路架構
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)      # 第一層卷積，輸入3通道，輸出6通道，kernel大小5
        self.pool = nn.MaxPool2d(2, 2)       # 最大池化層，2x2
        self.conv2 = nn.Conv2d(6, 16, 5)     # 第二層卷積，輸入6通道，輸出16通道
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 全連接層1
        self.fc2 = nn.Linear(120, 84)        # 全連接層2
        self.fc3 = nn.Linear(84, 10)         # 輸出層，10類別

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 卷積+ReLU+池化
        x = self.pool(F.relu(self.conv2(x))) # 卷積+ReLU+池化
        x = torch.flatten(x, 1)              # 展平成一維
        x = F.relu(self.fc1(x))              # 全連接+ReLU
        x = F.relu(self.fc2(x))              # 全連接+ReLU
        x = self.fc3(x)                      # 輸出層
        return x

net = Net()  # 建立網路物件



criterion = nn.CrossEntropyLoss()  # 使用交叉熵作為損失函數(rossEntropy(y,p)=− ∑ y log(p) )
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 使用 SGD 最佳化器

# 訓練網路
for epoch in range(2):  # 訓練2個 epoch
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()  # 歸零梯度
        outputs = net(inputs)  # 前向傳播
        loss = criterion(outputs, labels)  # 計算損失
        loss.backward()  # 反向傳播
        optimizer.step()  # 更新參數
        running_loss += loss.item()
        if i % 2000 == 1999:    # 每2000個 mini-batch 顯示一次損失
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
print('Finished Training')  # 訓練結束

# 儲存訓練好的模型
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# 取得一批測試影像並顯示
dataiter = iter(testloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

# 載入模型並進行預測
net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))
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

# 計算每個類別的準確率
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

# 設定運算裝置（GPU 或 CPU）
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)  # 將模型移到 GPU
inputs, labels = inputs.to(device), labels.to(device)  # 將資料移到 GPU
print(device)


del dataiter  # 釋放記憶體
