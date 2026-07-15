'''
#获取图像数据集
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()
#读数据集
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="./data", train=True, transform = trans,download=True)

mnist_test = torchvision.datasets.FashionMNIST(
    root="./data", train=False, transform = trans,download=True)
len(mnist_train),len(mnist_test)
mnist_train[0][0].shape
#将数据集放进一个DataLoader里，指定一个batch_size
def get_fashion_mnist_labels(labels):
    """返回数据集的文本标签"""
    text_labels = [
        't-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot'
    ]
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale = 1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols,figsize = figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
X,y = next(iter(data.DataLoader(mnist_train, batch_size = 18)))
show_images(X.reshape(18,28,28), 2, 9, titles=get_fashion_mnist_labels(y));

#小批量读取
batch_size = 256
def get_dataloader_workers():  #@save
    """#使用4个进程来读取数据"""
    return 4
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,num_workers=get_dataloader_workers())
timer = d2l.Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f} sec'
#整合组件
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

'''


#自己写代码实现
import torch
from IPython import display
from d2l import torch as d2l

# 初始化模型参数
num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# 定义 softmax 操作
def softmax(X):
    """手动实现 softmax 函数"""
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

# 定义模型
def net(X):
    """Softmax 回归模型"""
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

# 构建损失函数（交叉熵损失）
def cross_entropy(y_hat, y):
    """手动实现交叉熵损失函数"""
    return -torch.log(y_hat[range(len(y_hat)), y])

# 预测类别与真实 y 进行比较
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# 累加器类
class Accumulator:
    """在 n 个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 评估任意模型 net 的正确率
def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 训练模型一个迭代周期
def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期"""
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)  # 训练损失总和、训练准确度总和、样本数
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

# 画图类
class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

# 训练模型
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

# 预测函数
def predict_ch3(net, test_iter, n=6):
    """预测标签"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

# 使用示例（需要配合 Fashion-MNIST 数据集）
if __name__ == "__main__":
    # 设置学习率和训练轮数
    lr = 0.1
    num_epochs = 10

    # 定义更新器（使用 SGD 优化器）
    def updater(batch_size):
        return d2l.sgd([W, b], lr, batch_size)

    # 加载数据
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 训练模型
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

    # 预测结果
    predict_ch3(net, test_iter)

#用库
import torch
from torch import nn
from d2l import torch as d2l

# 单独设置批大小方便后续调整
batch_size = 256

# 直接用d2l中的load_data_fashion_mnist加载数据集
# 传入批大小参数,返回两个迭代器,第一个是训练数据集,第二个是测试数据集
# 都是可迭代对象,直接传入训练就可以
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 单独设置批大小方便后续调整
batch_size = 256

# 直接用d2l中的load_data_fashion_mnist加载数据集
# 传入批大小参数,返回两个迭代器,第一个是训练数据集,第二个是测试数据集
# 都是可迭代对象,直接传入训练就可以
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


# 定义一个权重初始化函数
def init_weights(m):
    # 只对线性层做初始化,其他层没有必要
    if type(m) == nn.Linear:
        # nn.init.normal_：PyTorch 提供的正态分布初始化函数
        # m.weight：线性层 m 中的权重矩阵
        nn.init.normal_(m.weight, std=0.01)


# net.apply(func)：PyTorch 中 nn.Module 的方法，
# 会递归地将函数 func 应用到网络 net 中的所有子模块.
net.apply(init_weights);
# 直接调用nn中的交叉熵函数
# reduction='none' 的核心作用是保留每个样本的独立损失值，
# 不做平均或求和，方便后续对损失进行精细化处理。
loss = nn.CrossEntropyLoss(reduction='none')
# 优化器直接用SGD,传入网络的参数和学习率
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
# 跑十个epoch
num_epochs = 10

# 调用d2l里面实现的一个函数
# 传入模型本身,训练集与测试集的迭代器,损失函数,循环次数与优化器
#d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

# 主程序入口
if __name__ == "__main__":
    # 超参数设置
    lr = 0.1
    num_epochs = 10
    batch_size = 256
    # SGD 更新器
    def updater(batch_size):
        return d2l.sgd([W, b], lr, batch_size)
    # 加载 Fashion-MNIST 数据集
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    # ✅ 调用本地手写的 train_ch3，不要加 d2l.
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    # 查看预测效果
    predict_ch3(net, test_iter)
