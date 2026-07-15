# 手动实现 Softmax 回归（不使用 PyTorch）
# 核心流程：softmax函数将得分转为概率 → 交叉熵损失衡量预测错误 → 梯度下降更新参数
#将样本x分为ABC三类
import numpy as np

#据集定义
#输入特征 X：4个样本，每个样本4个特征（4×4）
X = np.array([[2, 1, 3, 1],
              [3, 3, 4, 1],
              [4, 5, 1, 1],
              [5, 3, 2, 1]])
# 真实标签 y：每个样本对应的类别索引（0=A, 1=B, 2=C）
y = np.array([0, 1, 2, 1])
# 权重矩阵 w：4个特征 × 3个类别，初始值随机设定Y=XW+b
w = np.array([[0.1, 0.4, -0.5],   # 特征1对A/B/C类的权重
              [0.2, 0.7, -1.2],
              [1.2, 1.7, -1.5],
              [0.3, 0.6, -0.9]])
# 偏置向量 b（1×3）
b = np.array([0.1, 0.2, -0.1])

#2. Softmax函数
# 将原始得分 z 转换为概率分布，每个样本的概率之和为1
def softmax(z):
    # z - np.max(z, axis=1, keepdims=True)：减去每行最大值，防止指数运算数值溢出
    # np.exp()：对每个元素取指数
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    # np.sum(exp_z, axis=1, keepdims=True)：按行求和，得到每个样本所有类别的指数和
    sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)
    # 每个类别指数 / 该行指数和 = 该类别的概率
    return exp_z / sum_exp_z

#3. 交叉熵损失函数
#衡量预测概率分布与真实标签的差距，损失越小越好
def cross_entropy_loss(probs, y):
    n = len(y)  # 样本数量
    # probs[np.arange(n), y]：取每个样本真实标签对应的预测概率
    # + 1e-10：防止取对数时遇到0（log(0)无意义）
    # -np.log()：对概率取负对数，概率越小损失越大
    log_probs = -np.log(probs[np.arange(n), y] + 1e-10)
    # np.mean()：计算所有样本的平均损失
    return np.mean(log_probs)

#4. 梯度计算
# 根据链式法则，计算损失函数对权重 w 和偏置 b 的梯度
def compute_gradients(X, y, probs):
    n = len(y)  # 样本数量
    # 初始化梯度矩阵，复制概率矩阵
    dz = probs.copy()
    # softmax + 交叉熵的梯度公式简化：dL/dz = prob - y_onehot
    # 对每个样本，真实类别位置的概率减1，其他位置不变
    dz[np.arange(n), y] -= 1
    dz /= n  # 除以样本数，得到平均梯度
    # 对权重 w 的梯度：X的转置 × dz（形状：4×3）
    dw = X.T @ dz
    # 对偏置 b 的梯度：按列求和（形状：1×3）
    db = np.sum(dz, axis=0)
    return dw, db

#5. 训练过程
# 使用梯度下降法迭代更新参数，最小化损失函数
def train(X, y, w, b, lr=0.1, num_epochs=100):
    # 遍历训练轮数
    for epoch in range(num_epochs):
        #1前向传播：计算每个样本的原始得分 z = X·w + b
        z = X @ w + b
        #2将得分转为概率分布
        probs = softmax(z)
        #3计算当前损失
        loss = cross_entropy_loss(probs, y)
        #4反向传播：计算梯度
        dw, db = compute_gradients(X, y, probs)
        #5参数更新：沿梯度反方向调整参数
        w -= lr * dw
        b -= lr * db
        #每20轮打印一次损失，观察训练进度
        if (epoch + 1) % 20 == 0:
            print(f"训练轮数(epoch)：{epoch + 1}, 当前损失(loss):{loss:.6f}")
    # 返回训练好的参数
    return w, b


#6. 预测函数
# 使用训练好的参数对新样本进行预测
def predict(X, w, b):
    # 计算得分和概率
    z = X @ w + b
    probs = softmax(z)
    # np.argmax(probs, axis=1)：取概率最大的类别索引作为预测结果
    return np.argmax(probs, axis=1), probs


#7. 运行测试
print("初始预测（训练前）：")
z = X @ w + b
arrays = softmax(z)
typs = ['A', 'B', 'C']  # 类别名称映射
# 遍历每个样本，输出预测结果
for idx, array in enumerate(arrays):
    leibie = typs[np.argmax(array)]  # 预测类别名称
    max_prob = array[np.argmax(array)]  # 最大概率
    a, b_val, c = array  # 三个类别的概率
    print(f"样本{idx + 1}的类别为{leibie}，预测概率为{max_prob:.4f}，概率分布为[{a:.4f}, {b_val:.4f}, {c:.4f}]")

# 开始训练
print("\n开始训练...")
w_trained, b_trained = train(X, y, w, b, lr=0.5, num_epochs=100)  # 学习率0.5，训练100轮

# 训练后预测
print("\n训练后预测：")
preds, probs = predict(X, w_trained, b_trained)
# 遍历每个样本，输出训练后的预测结果
for idx, array in enumerate(probs):
    leibie = typs[preds[idx]]
    max_prob = array[preds[idx]]
    a, b_val, c = array
    print(f"样本{idx + 1}的类别为{leibie}，预测概率为{max_prob:.4f}，概率分布为[{a:.4f}, {b_val:.4f}, {c:.4f}]")