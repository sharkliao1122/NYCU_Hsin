# ReLU 激活 + Dropout
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#  ========== 設定運算裝置 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('目前運算裝置:', device)

# ========== 定義模型 ==========
class Net(nn.Module):
    def __init__(self, dropout_p=0.5):
        super(Net, self).__init__()
        # 四層卷積層
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=dropout_p)
        # 計算池化後的輸入維度：32x32 -> 2次池化後 8x8，通道數256
        self.fc1 = nn.Linear(256*8*8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        # 四層卷積 + ReLU + 池化
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout(x)
        # 展平成一維
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# ========== 數據集 ==========
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

classes = trainset.classes

# ========== 顏色設定 ==========
optimizer_colors = {
    "SGD": "red",
    "Adam": "blue",
    "RMSprop": "green"
}

# ========== 訓練與評估函式 ==========
def train_and_evaluate(model_class, model_name, dropout_p, optimizers, num_epochs=5, overfit_threshold=15):
    train_loss_history = {}
    train_acc_history = {}
    test_loss_history = {}
    test_acc_history = {}
    results = {}
    
    for opt_name, opt_fn in optimizers.items():
        print(f'\n===== {model_name} | Optimizer: {opt_name} =====')
        net = model_class(dropout_p=dropout_p).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = opt_fn(net)
        
        loss_list, acc_list = [], []
        test_loss_list, test_acc_list = [], []

        for epoch in range(num_epochs):
            # ---------- 訓練 ----------
            net.train()
            running_loss, correct, total = 0.0, 0, 0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
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

            # ---------- 測試 ----------
            net.eval()
            test_loss, test_correct, test_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
            avg_test_loss = test_loss / len(testloader)
            avg_test_acc = 100 * test_correct / test_total
            test_loss_list.append(avg_test_loss)
            test_acc_list.append(avg_test_acc)

            # ---------- 印出結果 ----------
            print(f'Epoch {epoch+1}: '
                  f'Train Loss={avg_loss:.4f}, Train Acc={avg_acc:.2f}% | '
                  f'Test Loss={avg_test_loss:.4f}, Test Acc={avg_test_acc:.2f}%')

            # ⚡ 自動過擬合提示
            acc_gap = avg_acc - avg_test_acc
            if acc_gap > overfit_threshold:
                print(f'⚠️ 注意：可能過擬合！(Train Acc 高 {acc_gap:.2f}% 比 Test Acc)')

        # 存歷史
        train_loss_history[opt_name] = loss_list
        train_acc_history[opt_name] = acc_list
        test_loss_history[opt_name] = test_loss_list
        test_acc_history[opt_name] = test_acc_list

        # ---------- 混淆矩陣 ----------
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        plt.title(f'Confusion Matrix ({model_name}, {opt_name}, ReLU)')
        print(f"[INFO] Saving confusion matrix: confusion_ReLU_{model_name}_{opt_name}.png")
        plt.savefig(f'C:\\Users\\s7103\\OneDrive\\桌面\\碩士班\\NYCU_Hsin\\week_3\\photo\\confusion_ReLU_{model_name}_{opt_name}.png')
        plt.close()

        # ---------- 誤分類圖片 ----------
        misclassified_images = []
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs, 1)
                for i in range(len(labels)):
                    if labels[i] != predicted[i]:
                        # 回到 CPU 以便顯示
                        misclassified_images.append((inputs[i].cpu(), predicted[i].cpu(), labels[i].cpu()))
                    if len(misclassified_images) >= 10:
                        break
                if len(misclassified_images) >= 10:
                    break

        plt.figure(figsize=(12, 6))
        n_img = len(misclassified_images)
        if n_img == 0:
            print(f"[WARNING] No misclassified images found for {model_name}, {opt_name}")
        for i, (img, pred, label) in enumerate(misclassified_images):
            plt.subplot(2, 5, i + 1)
            plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
            plt.title(f'P: {classes[pred]}\nT: {classes[label]}')
            plt.axis('off')
        # 若不足 10 張，補空白子圖
        for i in range(n_img, 10):
            plt.subplot(2, 5, i + 1)
            plt.axis('off')
        plt.suptitle(f'Misclassified Images ({model_name}, {opt_name}, LeakyReLU)')
        print(f"[INFO] Saving misclassified images: misclassified_LeakyReLU_{model_name}_{opt_name}.png")
        plt.savefig(f'C:\\Users\\s7103\\OneDrive\\桌面\\碩士班\\NYCU_Hsin\\week_3\\photo\\misclassified_ReLU_{model_name}_{opt_name}.png')
        plt.close()

        results[opt_name] = {
            'train_loss': loss_list,
            'train_acc': acc_list,
            'test_loss': test_loss_list,
            'test_acc': test_acc_list
        }

    # ---------- 畫圖：過擬合判斷 ----------
    plt.figure(figsize=(12,5))
    # Loss 曲線
    plt.subplot(1,2,1)
    for opt_name in train_loss_history.keys():
        color = optimizer_colors.get(opt_name, "black")
        plt.plot(train_loss_history[opt_name], label=f'{opt_name} (Train)', linestyle='-', color=color)
        plt.plot(test_loss_history[opt_name], label=f'{opt_name} (Test)', linestyle='--', color=color)
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0 , 3)
    plt.legend()

    # Accuracy 曲線
    plt.subplot(1,2,2)
    for opt_name in train_acc_history.keys():
        color = optimizer_colors.get(opt_name, "black")
        plt.plot(train_acc_history[opt_name], label=f'{opt_name} (Train)', linestyle='-', color=color)
        plt.plot(test_acc_history[opt_name], label=f'{opt_name} (Test)', linestyle='--', color=color)
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0 , 100)
    plt.legend()

    plt.tight_layout()
    print(f"[INFO] Saving overfitting curve: overfitting_curve_LeakyReLU_{model_name}.png")
    plt.savefig(f'C:\\Users\\s7103\\OneDrive\\桌面\\碩士班\\NYCU_Hsin\\week_3\\photo\\overfitting_curve_ReLU_{model_name}.png')
    print(f"[INFO] Saved overfitting curve: overfitting_curve_LeakyReLU_{model_name}.png")
    plt.close()
    return results

# ========== 測試執行 ==========
optimizers = {
    'SGD': lambda net: optim.SGD(net.parameters(), lr=0.01, momentum=0.9),
    'Adam': lambda net: optim.Adam(net.parameters(), lr=0.001),
    'RMSprop': lambda net: optim.RMSprop(net.parameters(), lr=0.001)
}

results = train_and_evaluate(Net, "FC_Net", dropout_p=0.5, optimizers=optimizers, num_epochs=25)
