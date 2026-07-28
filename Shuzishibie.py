"""
MNIST 手写数字识别 —— MLP 与 CNN 对比版
说明：原来是全连接 MLP，这里改为 CNN，并同时保留 MLP 作为对比，
用于观察哪一种在相同条件下更快、更准确。
"""

import os
import random
import ssl
import time

ssl._create_default_https_context = ssl._create_unverified_context

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


# ============================================================
# 1. 模型定义
# ============================================================
class MLPModel(nn.Module):
    """原始全连接网络: 784 -> 256 -> 128 -> 10"""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(128, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), -1)
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        return self.layer3(x)


class CNNModel(nn.Module):
    """卷积神经网络: 1x28x28 -> Conv -> Pool -> Conv -> Pool -> FC -> 10"""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ============================================================
# 2. 数据准备
# ============================================================
def mlp_preprocess(img) -> Tensor:
    return transforms.functional.to_tensor(img).view(-1)


def cnn_preprocess(img) -> Tensor:
    return transforms.functional.to_tensor(img)


train_dataset_mlp = torchvision.datasets.MNIST(
    root="./dataset",
    train=True,
    download=True,
    transform=mlp_preprocess,
)
test_dataset_mlp = torchvision.datasets.MNIST(
    root="./dataset",
    train=False,
    download=True,
    transform=mlp_preprocess,
)

train_dataset_cnn = torchvision.datasets.MNIST(
    root="./dataset",
    train=True,
    download=True,
    transform=cnn_preprocess,
)
test_dataset_cnn = torchvision.datasets.MNIST(
    root="./dataset",
    train=False,
    download=True,
    transform=cnn_preprocess,
)

train_loader_mlp = DataLoader(train_dataset_mlp, batch_size=64, shuffle=True)
test_loader_mlp = DataLoader(test_dataset_mlp, batch_size=64, shuffle=False)

train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=64, shuffle=True)
test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=64, shuffle=False)


# ============================================================
# 3. 训练与评估
# ============================================================
def evaluate(model, data_loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_model(model, train_loader, test_loader, model_path, epochs=2, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"✅ 已加载已有模型 → {model_path}")
        test_acc = evaluate(model, test_loader, device)
        return test_acc, 0.0

    print(f"⚠️ 未找到 {model_path}，开始训练...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total += images.size(0)

        avg_loss = total_loss / total
        print(f"  Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    test_acc = evaluate(model, test_loader, device)
    torch.save(model.state_dict(), model_path)
    print(f"✅ 训练完成，模型已保存 → {model_path}")
    return test_acc, elapsed


# ============================================================
# 4. 比较 MLP 与 CNN
# ============================================================
mlp_model = MLPModel()
cnn_model = CNNModel()

mlp_path = "Training_dataset/mnist_mlp_model.pth"
cnn_path = "Training_dataset/mnist_cnn_model.pth"

mlp_acc, mlp_time = train_model(mlp_model, train_loader_mlp, test_loader_mlp, mlp_path)
cnn_acc, cnn_time = train_model(cnn_model, train_loader_cnn, test_loader_cnn, cnn_path)

print("\n=== 结果对比 ===")
print(f"MLP 训练时间: {mlp_time:.2f}s，测试准确率: {mlp_acc:.4f}")
print(f"CNN 训练时间: {cnn_time:.2f}s，测试准确率: {cnn_acc:.4f}")

if cnn_time < mlp_time:
    print("✅ 结论：CNN 在当前设置下更快")
else:
    print("⚠️ 结论：MLP 在当前设置下更快")


# ============================================================
# 5. 预测与可视化（默认展示 CNN 的结果）
# ============================================================
img_idx = random.randint(0, len(test_dataset_cnn) - 1)
img, label = test_dataset_cnn[img_idx]

cnn_model.eval()
with torch.no_grad():
    logits = cnn_model(img.unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1).item()
    confidence = probs[0, pred].item()

print(f"随机选取图片索引: {img_idx}")
print(f"真实标签: {label},  预测结果: {pred},  置信度: {confidence:.4f}")

plt.imshow(img.squeeze(0).numpy(), cmap="gray")
plt.title(f"True: {label}  |  Pred: {pred}  ({confidence:.1%})")
plt.axis("off")
plt.show()
