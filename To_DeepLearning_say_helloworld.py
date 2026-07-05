print("hello world")

#pytorch安装输出测试
import torch                                      # 导入深度学习框架
print(torch.__version__)                         # 打印PyTorch版本，带+cu121代表GPU版
print(torch.version.cuda)                        # 打印框架绑定的CUDA版本
print(torch.cuda.is_available())                 # 判断NVIDIA显卡是否可加速，True=GPU生效
'''
hello world
2.4.1
12.1
True
'''