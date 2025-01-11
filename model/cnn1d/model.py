import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(CNN1D, self).__init__()
        self.l1 = nn.Conv1d(in_channel, 32, kernel_size, padding=1)   # 把out_channel变成32，features保持不变
        self.relu = nn.ReLU()
        self.l2 = nn.Conv1d(32, 64, kernel_size, padding=1)           # 把out_channel变成32，features保持不变
        self.l3 = nn.Conv1d(64, 32, kernel_size, padding=1)
        self.l4 = nn.Conv1d(32, 16, kernel_size, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 6, 32)    # fully connected layer，in_channel经过flatten变成16*feature的数量
        self.fc2 = nn.Linear(32, out_channel)    # fully connected layer，in_channel经过flatten变成16*feature的数量
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.l4(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    a = torch.rand((64, 1, 6))  # batch_size=64, in_channel=1, feature=6
    cnn1d = CNN1D(1, 1, 3)      # in_channel=1, out_channel=1, kernel size=3
    b = cnn1d(a)
    print(b.shape)              # 结果：torch.Size([64, 1])


