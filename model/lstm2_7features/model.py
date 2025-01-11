import torch
import torch.nn as nn

# MyLSTM2可以将先前的信息与当前的任务联系起来，但是由于在传导过程中容易出现梯度消失的问题，所以在实践中很难学习长距离依赖的数据(加上self-attention)。
class MyLSTM2(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(MyLSTM2, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm2 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True)   # batch_first=True代表lstm2模型的输入第一个参数为batch_size
        self.tanh = nn.Tanh()  # 在MyLSTM2中采用tanh是为了避免梯度消失或爆炸
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)  # 初始化成0
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        out, (hn, cn) = self.lstm2(x, (h0.detach(),c0.detach()))  # detach()函数用于从计算图中分离（detach）一个Tensor，创建一个新的Tensor，而不是在原始Tensor上进行操作。
        out = out[:, -1, :]
        out = self.fc(out)   # 第二个索引，取out第二维的最后一行数据
        return out


if __name__ == "__main__":
    a = torch.rand((64, 3, 6))  # batch_size=64, sequence length=3, input_size--features
    lstm2 = MyLSTM2(6, 64, 6, 1)      # input_dim=6, hidden_dim=64相当于neurons,6个进去变成了多少个, num_layers=6层LSTM, output_dim=1
    b = lstm2(a)
    print(b.shape)              # 结果：torch.Size([64, 1])


