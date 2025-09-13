import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D

# 步驟1：產生訓練資料
def dataset(show=True):
    # 產生 x 值，範圍從 -50 到 25，步長為 0.01
    x_1 = np.arange(-500, 500, 1)
    x_2 = np.arange(-500, 500, 1)
    x_3 = np.arange(-500, 500, 1)

    # 產生對應的 y 值，公式為 x_1 - x_2**2 + 100*x_3 + 4  並加入隨機雜訊
    y = x_1 - x_2**2 + 100*x_3 + 4 + np.random.randn(len(x_1)) * 1000

    # 如果 show 為 True，則繪製 3D 散佈圖
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x_1, x_2, x_3, c=y, cmap='viridis', alpha=0.5)
        ax.set_title('Data set')
        ax.set_xlabel('x_1')
        ax.set_ylabel('x_2')
        ax.set_zlabel('x_3')
        plt.show()

    # 回傳 x 與 y
    return (x_1, x_2, x_3), y   

# 呼叫 dataset 函數並取得 x, y
(x_1, x_2, x_3), y = dataset()
## 已在 dataset(show=True) 中繪製 3D 圖，這裡不需重複繪製

print("x_1 length:", x_1.shape)
print("x_2 length:", x_2.shape)
print("x_3 length:", x_3.shape)
print("y length:", y.shape)
print('x_1 type: ', type(x_1))
print('x_2 type: ', type(x_2))
print('x_3 type: ', type(x_3))
print('y type: ', type(y))
print('x_1 min, max: ', min(x_1), max(x_1))
print('x_2 min, max: ', min(x_2), max(x_2))
print('x_3 min, max: ', min(x_3), max(x_3))
print('y min, max: ', min(y), max(y))


# 將 x 正規化
x_1 = x_1 / max(x_1)
x_2 = x_2 / max(x_2)
x_3 = x_3 / max(x_3)
# 將 y 正規化
y = y / max(y)

print(min(x_1), max(x_1 ))
print(min(x_2), max(x_2 ))
print(min(x_3), max(x_3 ))
print(min(y), max(y))

# 將 x 轉為 float 型態的 tensor，並增加一個維度
x_1_torch = torch.tensor(x_1, dtype=torch.float).unsqueeze(1)
x_2_torch = torch.tensor(x_2, dtype=torch.float).unsqueeze(1)
x_3_torch = torch.tensor(x_3, dtype=torch.float).unsqueeze(1)

# 將 y 轉為 float 型態的 tensor，並增加一個維度
y_torch = torch.tensor(y, dtype=torch.float).unsqueeze(1)

print('x_1 型態: ', type(x_1_torch))
print('x_2 型態: ', type(x_2_torch))
print('x_3 型態: ', type(x_3_torch))
print('y 型態: ', type(y_torch))
print(x_1_torch.shape)
print(x_2_torch.shape)
print(x_3_torch.shape)
print(y_torch.shape)


## 步驟2：定義模型架構
# 建立一個簡單的線性迴歸模型（單一線性層）
model = torch.nn.Sequential(torch.nn.Linear(in_features=3, out_features=1))

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
    X = torch.cat([x_1_torch, x_2_torch, x_3_torch], dim=1)
    y_pred = model(X)
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
X = torch.cat([x_1_torch, x_2_torch, x_3_torch], dim=1)
y_hat = model(X).detach().numpy()

# 步驟11：視覺化原始資料、預測線與 MSE

# 3D scatter plot: 原始資料點 (藍色) 與預測點 (紅色)
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')
# 原始資料點：藍色圓形
sc1 = ax.scatter(x_1, x_2, x_3, c=y, cmap='Blues', label='Data', alpha=0.5, marker='o')
# 預測資料點：紅色三角形
sc2 = ax.scatter(x_1, x_2, x_3, c=y_hat.flatten(), cmap='Reds', label='Predicted', alpha=0.5, marker='^')
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
ax.set_zlabel('x_3')
ax.text(-1.8, 1, 1, 'MSE = {:.3f}'.format(mse), fontsize=15)
ax.grid(True)
ax.legend(fontsize=20)
ax.set_title('Pytorch Model')
plt.show()