import torch
import torch.nn as nn

# LSTM可以将先前的信息与当前的任务联系起来，但是由于在传导过程中容易出现梯度消失的问题，所以在实践中很难学习长距离依赖的数据(加上self-attention)。
# GRU的出现：LSTM训练参数多，较复杂，容易过拟合；而GRU结构较简单，参数减少，逐渐流行。
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        # Define attention layers三个线性层（全连接层）用于将lstm_output投影到三个不同的空间：查询空间、键空间和值空间。这些投影是计算注意力分数所必需的。
        self.query_layer = nn.Linear(hidden_dim, hidden_dim)
        self.key_layer = nn.Linear(hidden_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)  # 对注意力分数应用softmax函数，从而得到注意力权重，确保它们在输入序列中的每个标记上加起来等于1。

    def forward(self, lstm_output):
        # Compute query, key, and value
        query = self.query_layer(lstm_output)
        key = self.key_layer(lstm_output)
        value = self.value_layer(lstm_output)

        # Compute attention scores
        # 对查询张量和键张量的转置执行矩阵乘法，计算注意力分数。这个操作度量了序列中每个标记与其他标记的相似性，生成了注意力分数。
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        # 注意力分数被缩放，通过将其除以隐藏维度的平方根。这种缩放是自注意力机制中的常见做法，可以避免出现大的值，并稳定训练过程中的梯度。
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.hidden_dim).float())

        # Apply softmax to get attention weights
        # 缩放后的注意力分数被应用softmax函数，得到注意力权重。这确保了每个标记相对于其他标记的重要性或相关性，并且每个注意力权重的总和等于1。
        attention_weights = self.softmax(attention_scores)

        # Compute the weighted sum of the value vectors
        # 通过矩阵乘法将注意力权重与值张量相乘，计算得到加权和的值向量。
        attention_output = torch.matmul(attention_weights, value)

        return attention_output # 返回注意力输出（即加权和的值向量），这就是自注意力层的输出。
class MyLSTM2_ATN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(MyLSTM2_ATN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # self.lstm2_atn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True)   # batch_first=True代表LSTM模型的输入第一个参数为batch_size
        self.lstm2_atn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first = True)   # batch_first=True代表LSTM模型的输入第一个参数为batch_size
        self.attention = SelfAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        # self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)  # 初始化成0
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        # out, (hn, cn) = self.lstm2_atn(x, (h0.detach(),c0.detach()))  # detach()函数用于从计算图中分离（detach）一个Tensor，创建一个新的Tensor，而不是在原始Tensor上进行操作。
        out, _ = self.lstm2_atn(x, h0.detach())  # detach()函数用于从计算图中分离（detach）一个Tensor，创建一个新的Tensor，而不是在原始Tensor上进行操作。
        out = self.attention(out)
        out = out[:, -1, :]    # 第二个索引，取out第二维的最后一行数据
        out = self.fc(out)
        return out


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a = torch.rand((32, 3, 6)).to(device)  # batch_size=64, sequence length=3, input_size--features
    lstm2_atn = MyLSTM2_ATN(6, 64, 6, 1).to(device)      # input_dim=6, hidden_dim=64相当于neurons,6个进去变成了多少个, num_layers=6层LSTM, output_dim=1
    b = lstm2_atn(a)
    print(b.shape)              # 结果：torch.Size([64, 1])


