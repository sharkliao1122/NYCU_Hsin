
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F

# 步驟1：產生訓練資料
def dataset(show=True):
    # 產生 x 值，範圍從 -25 到 25，步長為 0.01
    x = np.arange(-25, 25, 0.01)

    # 產生對應的 y 值，公式為 x^3 + 20 並加入隨機雜訊
    y = x**3 + 20 + np.random.randn(len(x)) * 1000

    # 如果 show 為 True，則繪製散佈圖
    if show:
        plt.scatter(x, y)
        plt.show()

    # 回傳 x 與 y
    return x, y

# 呼叫 dataset 函數並取得 x, y
x, y = dataset()

print("x 長度:", x.shape)
print("y 長度:", y.shape)
print('x 型態: ', type(x))
print('y 型態: ', type(y))
print('x 最小值, 最大值: ', min(x), max(x))
print('y 最小值, 最大值: ', min(y), max(y))

# 將 x 正規化
x = x / max(x)
# 將 y 正規化
y = y / max(y)

print(min(x), max(x))
print(min(y), max(y))

# 將 x 轉為 float 型態的 tensor，並增加一個維度
x_torch = torch.tensor(x, dtype=torch.float).unsqueeze(1)
# 將 y 轉為 float 型態的 tensor，並增加一個維度
y_torch = torch.tensor(y, dtype=torch.float).unsqueeze(1)

print('x 型態: ', type(x_torch))
print('y 型態: ', type(y_torch))
print(x_torch.shape)
print(y_torch.shape)

## 步驟2：定義模型架構
# 建立一個簡單的線性迴歸模型（單一線性層）
model = torch.nn.Sequential(torch.nn.Linear(in_features=1, out_features=1))

## 步驟3：設定超參數
# 設定學習率
learning_rate = 1e-3

## 步驟4：定義損失函數
# 使用均方誤差（MSE）作為回歸任務的損失函數
loss_func = torch.nn.MSELoss(reduction='mean')

## 步驟5：定義優化器
# 使用 Adam 優化器更新模型參數
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

## 步驟6：設定訓練回合數
num_ep = 500

## 步驟7：建立損失記錄清單
loss_history = []

## 步驟8：訓練迴圈
for ep in range(num_ep):
    # 前向傳播，預測 y
    y_pred = model(x_torch)
    # 計算損失
    loss = loss_func(y_pred, y_torch)
    # 記錄損失
    loss_history.append(loss.item())
    # 清除梯度
    optimizer.zero_grad()
    # 反向傳播
    loss.backward()
    # 更新參數
    optimizer.step()
    # 繪製損失曲線（可選）
    plt.plot(loss_history)

# 步驟9：繪製損失曲線
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.show()

# 步驟10：計算最終均方誤差並進行預測
mse = loss_history[-1]
# 使用訓練好的模型預測
y_hat = model(x_torch).detach().numpy()

# 步驟11：視覺化原始資料、預測線與 MSE
plt.figure(figsize=(12, 7))
plt.title('PyTorch Model')
plt.scatter(x, y, label='Data $(x, y)$')
plt.plot(x, y_hat, color='red', label='Predicted Line $y = f(x)$', linewidth=4.0)
plt.xlabel('$x$', fontsize=20)
plt.ylabel('$y$', fontsize=20)
plt.text(0, 0.70, 'MSE = {:.3f}'.format(mse), fontsize=20)
plt.grid(True)
plt.legend(fontsize=20)
plt.show()

# 定義無激活函數的神經網路類別
class Net_no_activation(nn.Module):
    def __init__(self, hidden_size):
        super(Net_no_activation, self).__init__()
        self.hidden_size = hidden_size
        self.layer1 = nn.Linear(1, self.hidden_size)  # 第一層線性層
        self.layer2 = nn.Linear(self.hidden_size, 1)  # 第二層線性層

    def forward(self, x):
        x = self.layer1(x)  # 第一層
        # x = F.relu(x)      # 已註解掉的激活函數
        x = self.layer2(x)  # 第二層
        return x

# 建立無激活函數神經網路實例，隱藏層大小 100
multi_layer_model_no_activation = Net_no_activation(hidden_size=100)

# 設定學習率
learning_rate = 1e-3

# 定義損失函數
loss_func = torch.nn.MSELoss(reduction='mean')

# 定義優化器
multi_layer_optimizer_no_activation = torch.optim.Adam(multi_layer_model_no_activation.parameters(), lr=learning_rate)

# 訓練回合數
num_ep = 1000

# 建立損失記錄清單
loss_history = []

# 訓練迴圈
for ep in range(num_ep):
    y_pred = multi_layer_model_no_activation(x_torch)
    loss = loss_func(y_pred, y_torch)
    loss_history.append(loss.item())
    multi_layer_optimizer_no_activation.zero_grad()
    loss.backward()
    multi_layer_optimizer_no_activation.step()

# 繪製損失曲線
plt.plot(loss_history)
plt.xlabel('Epochs')  # 訓練回合
plt.ylabel('MSE Loss') # 均方誤差
plt.show()

# 最終均方誤差
mse = loss_history[-1]
# 預測
y_hat = multi_layer_model_no_activation(x_torch).detach().numpy()

# 視覺化
plt.figure(figsize=(12, 7))
plt.title('PyTorch Model')
plt.scatter(x, y, label='Data $(x, y)$')
plt.plot(x, y_hat, color='red', label='Predicted Line $y = f(x)$', linewidth=4.0)
plt.xlabel('$x$', fontsize=20)
plt.ylabel('$y$', fontsize=20)
plt.text(0, 0.70, 'MSE = {:.3f}'.format(mse), fontsize=20)
plt.grid(True)
plt.legend(fontsize=20)
plt.show()

# 定義有激活函數的神經網路類別
class Net(nn.Module):
    def __init__(self, hidden_size):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.layer1 = nn.Linear(1, self.hidden_size)  # 第一層線性層
        self.layer2 = nn.Linear(self.hidden_size, 1)  # 第二層線性層

    def forward(self, x):
        x = self.layer1(x)        # 第一層
        x = F.relu(x)             # ReLU 激活函數
        x = self.layer2(x)        # 第二層
        return x

# 建立有激活函數神經網路實例，隱藏層大小 100
multi_layer_model = Net(hidden_size=100)

# 設定學習率
learning_rate = 1e-3

# 定義損失函數
loss_func = torch.nn.MSELoss(reduction='mean')

# 定義優化器
multi_layer_optimizer = torch.optim.Adam(multi_layer_model.parameters(), lr=learning_rate)

# 訓練回合數
num_ep = 500

# 建立損失記錄清單
loss_history = []

# 訓練迴圈
for ep in range(num_ep):
    y_pred = multi_layer_model(x_torch)
    loss = loss_func(y_pred, y_torch)
    loss_history.append(loss.item())
    multi_layer_optimizer.zero_grad()
    loss.backward()
    multi_layer_optimizer.step()

# 繪製損失曲線
plt.plot(loss_history)
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.show()

# 最終均方誤差
mse = loss_history[-1]
# 預測
y_hat = multi_layer_model(x_torch).detach().numpy()

# 視覺化
plt.figure(figsize=(12, 7))
plt.title('PyTorch Model')
plt.scatter(x, y, label='Data $(x, y)$')
plt.plot(x, y_hat, color='red', label='Predicted Line $y = f(x)$', linewidth=4.0)
plt.xlabel('$x$', fontsize=20)
plt.ylabel('$y$', fontsize=20)
plt.text(0, 0.70, 'MSE = {:.3f}'.format(mse), fontsize=20)
plt.grid(True)
plt.legend(fontsize=20)
plt.show()