import torch
import torch.nn as nn

# 定义了一个名为 MLP 的类，并继承了 nn.Module 类。
# nn.Module 是 PyTorch 中用于构建神经网络模型的基类，通过继承它，我们可以使用其中定义的功能和属性来构建自定义的神经网络模型。
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        # 调用父类 nn.Module 的构造函数，确保继承自 nn.Module 的属性和方法正确初始化。
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)  # 定义了input layer，输入维度为 input_dim，输出维度为 hidden_dim。
        self.l2 = nn.Linear(hidden_dim, hidden_dim) # 第一个hidden layer
        self.l3 = nn.Linear(hidden_dim, hidden_dim) # 第二个hidden layer
        self.l4 = nn.Linear(hidden_dim, output_dim) # 定义了output layer，输入维度为 hidden_dim，输出维度为 output_dim。
        self.relu = nn.ReLU()          # 定义激活函数ReLU
        self.tanh = nn.Tanh()          # 定义激活函数Tanh
        self.sigmoid = nn.Sigmoid()    # 定义激活函数Sigmoid


    def fun2(self, x):   # 这是 MLP 类中的一个方法，用于自定义的函数 fun2，可以根据需要在模型中添加自己的功能。
        return x ** 2

    def forward(self, x):  # 这是 MLP 类中的前向传播方法，用于定义模型的计算流程。
        # 在 PyTorch 中，神经网络模型的前向传播是通过调用模型的 forward() 方法来实现的。
        # 而 fun2() 方法只是一个自定义的函数，并不会自动在实例运行时被调用。
        # 如果要用fun2：
        # 在forward方法下添加：x = self.fun2(x) 或者
        # 在类之外手工调用：model = MLP(input_dim, output_dim, hidden_dim)
        #                 result = model.fun2(some_input)

        x = self.l1(x)     # x经过l1
        x = self.relu(x)   # x经过激活函数
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.l4(x)
        return x


if __name__ == "__main__":
    a = torch.rand((64, 6))  # 随机生成64*6的矩阵（64行是一个batch，有6个features)
    mlp = MLP(6, 1, 32)      # 分别对MLP的三个dimension进行赋值，因为l1是linear的，所以input_dim要等于上一个function的最后一个参数
    b = mlp(a)
    print(b.shape)           # 结果：torch.Size([64, 1])
    # a = torch.rand((64,10,6))    # 随机生成64*10*6的矩阵，相当于一个三维的空间，一共64层，最后一个参数等于MLP第一个参数就行
    # mlp = MLP(6, 1, 32)
    # b = mlp(a)
    # print(b.shape)               # 结果：torch.Size([64, 10, 1])
    # 在神经网络中，"batch" 是指一次训练中同时处理的样本数量。通常，将训练数据划分为多个小的批次（batches）进行模型的训练。

