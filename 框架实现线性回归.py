# 导入依赖库
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn  # 导入神经网络模块


#  生成合成数据集
# 定义真实线性参数
true_w = torch.tensor([2, -3.4])
true_b = 4.2
# 调用d2l库函数生成带噪声的线性数据集：y = Xw + b + 噪声
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

#读取数据集
# 2. 构造数据迭代器
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器
    参数：
        data_arrays: 输入张量元组 (features, labels)
        batch_size: 每批次样本数量
        is_train: 是否为训练集，True则打乱数据顺序
    """
    # 将特征和标签打包为TensorDataset数据集
    dataset = data.TensorDataset(*data_arrays)
    # 返回DataLoader迭代器，实现小批量加载数据
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

# 设置批次大小
batch_size = 10
# 创建数据迭代器
data_iter = load_array((features, labels), batch_size)
# 查看第一批数据
next(iter(data_iter))

#定义模型
# 3. 搭建线性回归模型
# 使用Sequential容器构建单层线性模型：输入2维特征，输出1维预测值
net = nn.Sequential(nn.Linear(2, 1))
# 初始化模型权重参数：正态分布初始化(均值0，标准差0.01)
net[0].weight.data.normal_(0, 0.01)
# 初始化偏置参数：全部填充为0
net[0].bias.data.fill_(0)


# 4. 定义损失函数
# 均方误差损失函数 MSELoss（平方损失）
loss = nn.MSELoss()

#定义优化算法
# 5. 定义优化器SGD
# 实例化随机梯度下降优化器，学习率lr=0.03，更新net模型参数
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

#训练
#6. 模型主训练循环
num_epochs = 3  # 设置训练总轮次
for epoch in range(num_epochs):
    # 遍历每一个小批量数据
    for X, y in data_iter:
        l = loss(net(X), y)        # 前向传播：计算当前批次预测损失
        trainer.zero_grad()       # 清空优化器累计梯度，防止梯度叠加
        l.backward()               # 反向传播：自动计算参数梯度
        trainer.step()             # SGD梯度下降更新模型参数
    # 每轮epoch结束后，计算全量数据集上的损失
    l = loss(net(features), labels)
    # 打印本轮训练结果
    print(f'epoch {epoch + 1}, loss {l:f}')
