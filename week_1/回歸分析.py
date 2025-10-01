import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import os

# 步驟1：產生訓練資料
def dataset(show=True):
    
    # 產生 x 值，範圍從 -50 到 500，步長為 0.01
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

# 依題目參數設定
def run_case(n_hidden, beta):
    
    # 產生資料
    (x_1, x_2, x_3), y = dataset(show=False)
    
    # 正規化
    x_1 = x_1 / max(x_1)
    x_2 = x_2 / max(x_2)
    x_3 = x_3 / max(x_3)
    y = y / max(y)
    
    # 組合成 observation matrix (N, 3)
    X = np.stack([x_1, x_2, x_3], axis=1)
    
    # 分割訓練/測試資料
    N = X.shape[0]
    idx = int(N * beta)
    X_train, X_test = X[:idx], X[idx:]
    y_train, y_test = y[:idx], y[idx:]
    
    # 轉為 tensor
    X_train_torch = torch.tensor(X_train, dtype=torch.float)
    y_train_torch = torch.tensor(y_train, dtype=torch.float).unsqueeze(1)
    X_test_torch = torch.tensor(X_test, dtype=torch.float)
    y_test_torch = torch.tensor(y_test, dtype=torch.float).unsqueeze(1)
    
    # 建立模型：一層隱藏層
    model = nn.Sequential(
        nn.Linear(3, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, 1)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.MSELoss()
    
     # 建立儲存圖片的資料夾（如果不存在）
    chart_dir = os.path.join('.', 'week_2_chart_linear')
    os.makedirs(chart_dir, exist_ok=True)

    # 訓練
    loss_history = []
    for ep in range(500):
        y_pred = model(X_train_torch)
        loss = loss_func(y_pred, y_train_torch)
        loss_history.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    plt.plot(loss_history)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve: n_hidden={n_hidden}, beta={beta}')
    plt.savefig(os.path.join(chart_dir, f'losscurve_n{n_hidden}_beta{beta}.png')) 
    plt.close()
    
    # 預測
    y_train_hat = model(X_train_torch).detach().numpy().flatten()
    y_test_hat = model(X_test_torch).detach().numpy().flatten()
    
    # 計算 MSE
    train_mse = np.mean((y_train_hat - y_train)**2)
    test_mse = np.mean((y_test_hat - y_test)**2)
    
    
    
    
    # 畫測試資料
    plt.figure(figsize=(7,7))
    plt.scatter(y_test, y_test_hat, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('True y (test)')
    plt.ylabel('Predicted y_hat (test)')
    plt.title(f'Test: n={n_hidden}, beta={beta}, MSE={test_mse:.4f}')
    plt.savefig(os.path.join(chart_dir, f'test_n{n_hidden}_beta{beta}.png'))
    
    # 回傳結果
    return train_mse, test_mse
    
    # 回傳結果
    return y_train, y_train_hat, y_test, y_test_hat

# 題目要求的五種 case
cases = [
    (1, 0.7),
    (2, 0.7),
    (100, 0.7),
    (2, 0.3),
    (100, 0.3)
]
results = []
for n_hidden, beta in cases:
    print(f'Running case: n={n_hidden}, beta={beta}')
    train_mse, test_mse = run_case(n_hidden, beta)
    results.append((n_hidden, beta, train_mse, test_mse))

print("\n各 case 的 MSE：")
for i, (n_hidden, beta, train_mse, test_mse) in enumerate(results, 1):
    print(f"Case {i}: n={n_hidden}, beta={beta} | Train MSE={train_mse:.4f} | Test MSE={test_mse:.4f}")

