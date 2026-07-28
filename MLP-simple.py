import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs
        # He/Kaiming 初始化（适配ReLU），避免梯度消失
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * (2.0 / num_inputs) ** 0.5)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * (2.0 / num_hiddens) ** 0.5)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))

    # 定义前向传播
    def forward(self, X):
        X = X.reshape((-1, self.num_inputs))
        H = self.relu(X @ self.W1 + self.b1)
        return H @ self.W2 + self.b2

    # 定义ReLU激活函数
    def relu(self, X):
        return torch.max(X, torch.zeros_like(X))


# 单轮训练逻辑（替代d2l.train_epoch_ch3）
def train_epoch(net, train_iter, loss, optimizer):
    net.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for X, y in train_iter:
        # 前向传播
        y_hat = net(X)
        l = loss(y_hat, y)
        # 反向传播更新参数
        optimizer.zero_grad()
        l.mean().backward()
        optimizer.step()
        # 统计损失与精度
        total_loss += l.sum().item()
        preds = torch.argmax(y_hat, dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += y.numel()
    train_loss = total_loss / total_samples
    train_acc = total_correct / total_samples
    return train_loss, train_acc


# 评估测试集精度（替代d2l.evaluate_accuracy）
def evaluate_acc(net, test_iter):
    net.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, y in test_iter:
            y_hat = net(X)
            preds = torch.argmax(y_hat, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.numel()
    return total_correct / total_samples


# 完整训练流程（替代d2l.train_ch3）
def train(net, train_iter, test_iter, loss, num_epochs, optimizer):
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_iter, loss, optimizer)
        test_acc = evaluate_acc(net, test_iter)
        animator.add(epoch + 1, (train_loss, train_acc, test_acc))
    print(f'最终训练损失: {train_loss:.3f}, 训练精度: {train_acc:.3f}, 测试精度: {test_acc:.3f}')


# 手动实现预测可视化（替代d2l.predict_ch3）
def predict_show(net, test_iter, n=8):
    net.eval()
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    with torch.no_grad():
        for X, y in test_iter:
            y_hat = net(X)
            preds = torch.argmax(y_hat, dim=1)
            break
    # 转到CPU便于matplotlib绘图
    images = X[:n].cpu().reshape(n, 28, 28)
    trues = y[:n].cpu()
    pred_res = preds[:n].cpu()
    fig, axes = plt.subplots(1, n, figsize=(14, 2.5))
    for i in range(n):
        axes[i].imshow(images[i], cmap='gray')
        # 正确预测用绿色标题，错误预测用红色标题
        color = 'green' if trues[i] == pred_res[i] else 'red'
        axes[i].set_title(f'真实: {labels[trues[i]]}\n预测: {labels[pred_res[i]]}',
                          color=color, fontsize=9)
        axes[i].axis('off')
    plt.tight_layout()
    fig.suptitle('绿色=正确  红色=错误', fontsize=12, y=1.02)


if __name__ == '__main__':
    # 设置超参数
    batch_size = 256
    num_inputs, num_hiddens, num_outputs = 784, 256, 10
    # 加载数据
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    # 实例网络
    net = Net(num_inputs, num_hiddens, num_outputs)
    # 损失
    loss = nn.CrossEntropyLoss(reduction='none')
    # 训练配置
    num_epochs, lr = 10, 0.1
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    # 训练
    train(net, train_iter, test_iter, loss, num_epochs, optimizer)
    plt.show()

    # 预测绘图
    predict_show(net, test_iter)
    plt.show()
