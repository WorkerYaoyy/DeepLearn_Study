import torch

x=torch.tensor([[1,2],[3,4],[1,3]])
x=torch.tensor([[3,2],[2,4],[5,3]])

#输出多少行多少列
print(x.size())
print(x.shape)
#
# 当 CUDA 可用的时候，可用运行下方这段代码，采用 torch.device() 方法来改变 tensors 是否在 GPU 上进行计算操作
# 1. 判断本机CUDA(GPU)环境是否可用
if torch.cuda.is_available():
    # 2. 创建设备对象：cuda 代表GPU
    device = torch.device("cuda")
    # 3. 参照x的形状，创建全1张量y，直接分配在GPU显存中
    y = torch.ones_like(x, device=device)
    # 4. 将原本在CPU上的张量x，迁移复制到GPU上
    x = x.to(device)
    # 5. GPU上执行加法运算：x + y，结果z依旧存放在GPU
    z = x + y
    # 6. 打印GPU上的张量z
    print(z)
    # 7. 把z从GPU拷贝回CPU内存，同时数据类型转为 float64(double)，然后打印
    print(z.to("cpu", torch.double))
"""
1. 自动微分 autograd
PyTorch 依靠 autograd 自动计算张量梯度（导数），训练神经网络反向传播底层全靠它。
2. torch.autograd.grad(outputs, inputs) 作用
手动计算标量 / 张量 outputs 对 inputs 的导数，返回梯度张量，不修改原张量 .grad 属性。

torch.autograd.grad(
    outputs,        # 需要求导的目标（标量/张量）
    inputs,         # 对谁求导，张量/张量列表
    grad_outputs=None, # 链式求导上游梯度（多输出必备）
    create_graph=False, # True=生成二阶导计算图（求高阶导数）
    retain_graph=False, # True=保留计算图，可多次求导
    only_inputs=True
)
"""
