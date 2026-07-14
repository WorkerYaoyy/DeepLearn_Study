#手动实现 线性回归 + 小批量随机梯度下降（SGD）
#从带噪声的合成数据里学习逼近真实参数 `true_w=[2, -3.4]`、`true_b=4.2`，完成模型拟合训练。
import random
import torch

#生成数据集
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声。"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
# 给定参数
true_w = torch.tensor([2, -3.4])
true_b = 4.2
#features：输入特征（1000×2）；labels：对应标签（1000×1）
#一共 1000 条样本，每条样本有 2 个输入特征，搭配 1 个真实标签，是监督学习最基础的数据集格式。
#f(x1,x2)→y
features, labels = synthetic_data(true_w, true_b, 1000)
"""打印显示训练集数据
print('features:', features[0], '\nlabel:', labels[0])
d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(),
                labels.detach().numpy(), 1);
"""

#读取数据集
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices) #打乱样本索引，实现随机采样
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
batch_size = 10
'''
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
'''

#定义初始化模型和初始化参数
w=torch.normal(0,0.01,size=(2,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)
def linreg(X,w,b):
    """线性回归模型"""
    return torch.matmul(X,w)+b#矩阵乘法

#定义损失函数
#作用：计算预测值 y^2和真实标签 y 的误差；reshape 保证两个张量维度一致，避免形状报错
def square_loss(y_hat,y):
    '''均方损失'''
    return (y_hat-y.reshape(y_hat.shape))**2/2  #**是平方

#定义优化算法
#用线性回归模型的梯度下降公式更新参数
def sgd(params, lr, batch_size):
    """小批量随机梯度下降。"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size#梯度下降更新
            param.grad.zero_()

#训练过程
lr = 0.03
num_epochs = 3
net = linreg
loss = square_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # `X`和`y`的小批量损失
        # 因为`l`形状是(`batch_size`, 1)，而不是一个标量。`l`中的所有元素被加到
        # 并以此计算关于[`w`, `b`]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
'''
- 超参数：学习率 `lr=0.03`，总共遍历数据集 `num_epochs=3` 轮
- 内层循环：遍历每一个小批量
  1. 前向传播：用当前模型预测，计算批次损失
  2. `l.sum().backward()`：自动微分，计算 w、b 的梯度
  3. 调用 SGD 更新参数
- 每轮 epoch 结束后：在**全量数据集**上计算整体损失并打印，观察损失下降收敛情况
- 预期结果：损失持续下降，最终训练得到的 w、b 会非常接近真实值 `[2,-3.4]` 和 `4.2`
'''