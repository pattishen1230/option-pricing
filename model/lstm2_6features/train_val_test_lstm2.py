import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from model import MyLSTM2

class Feeder(torch.utils.data.Dataset):
    def __init__(self, data_path, train_val_test='train'):
        data = pd.read_csv(data_path)
        features = data.iloc[:, [i for i in range(18)]]
        features = features.to_numpy()  # 将 DataFrame 转换为 NumPy 数组再进行归一化，因为许多归一化函数或库（如 Scikit-learn 中的 MinMaxScaler）接受的输入类型是 NumPy 数组，而不是 DataFrame。当然了归一化的对象必须是数值。
        features = features.astype(np.float32)  # [N, 18] -- > [N, 3, 6]
        # normalization(使所有特征对最后结果影响的权重相同，避免梯度消失或爆炸）
        f_min = np.min(features, axis=0)  # 计算features每列的最小值
        f_max = np.max(features, axis=0)
        features = (features - f_min) / (f_max - f_min)
        features = features.reshape(features.shape[0], 3, 6)
        features = features[:, ::-1, :]   # 将中间维度的数据反向排列（之前是t t-1 t-2的数据，现在改为t-2,t-1,t)
        labels = data.iloc[:, [18]]
        labels = labels.to_numpy()
        labels = labels.astype(np.float32)
        # 划分训练，验证和测试集
        total_num = features.shape[0]

        # 打乱索引的顺序，然后features和labels按照索引取值
        permutation = np.random.permutation(total_num)
        features, labels = features[permutation], labels[permutation]

        train_idx_list = list(np.arange(0, int(total_num*0.7)))  # 获取训练集的索引的数字list（0，总行数的0.7倍）
        val_idx_list = list(np.arange(int(total_num*0.7), int(total_num*0.8)))
        test_idx_list = list(np.arange(int(total_num*0.8), total_num))
        if train_val_test.lower() == 'train':   # 不管大小写，都变成小写
            self.features = features[train_idx_list]    # self.features：提取features里面的行（如果参数是'train'）
            self.labels = labels[train_idx_list]
        elif train_val_test.lower() == 'val':
            self.features = features[val_idx_list]
            self.labels = labels[val_idx_list]
        elif train_val_test.lower() == 'test':
            self.features = features[test_idx_list]
            self.labels = labels[test_idx_list]
        else:
            raise Exception("参数train_val_test定义错误！")   # 如果参数不是以上三个数据集就报错

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def set_random_seed(seed=42):
    # 设置PyTorch的随机种子
    torch.manual_seed(seed)
    # 设置CUDA的随机种子（如果使用GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    # 设置Python标准库的随机种子
    random.seed(seed)
    # 设置Numpy的随机种子
    np.random.seed(seed)

if __name__=="__main__":
    set_random_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(">>>>>>>>>>>> Training on ", device)
    # 训练和验证模型，把验证集上表现最好的模型保存起来
    writer = SummaryWriter('./logs_lstm2')
    print(os.path.dirname(os.path.dirname(os.getcwd())))
    root_path = os.path.dirname(os.path.dirname(os.getcwd())) + '/data/v1.2'
    files = os.listdir(root_path)
    print(files)
    for file in files:
        if file == ".DS_Store" or file != 'op_money_1.03dte_180.csv':
            continue
        print("Processing %s ..." % file)
        data_path = os.path.join(root_path, file)
        train_feeder = Feeder(data_path, train_val_test='train')
        train_loader = torch.utils.data.DataLoader(dataset=train_feeder, batch_size=1024, shuffle=True)
        val_feeder = Feeder(data_path, train_val_test='val')
        val_loader = torch.utils.data.DataLoader(dataset=val_feeder, batch_size=1024, shuffle=True)
        print("Length of training set: %d ..." % len(train_feeder))
        # define a MyLSTM2
        lstm2 = MyLSTM2(6, 64, 6, 1).to(device)
        # lstm2.load_state_dict(torch.load('./lstm2_best_models_final/lstm2_best_val_'+ file + '.pth'))
        optim = Adam(lstm2.parameters(), lr=0.001)

        train_total_loss = []
        val_total_loss = []
        loss = nn.MSELoss()
        best_val_loss = np.float32('inf')
        best_epoch = 0
        for epoch in range(2000):
            running_loss = 0
            lstm2.train()
            for data in train_loader:
                train_feature, train_label = data
                train_feature = train_feature.to(device)
                train_label = train_label.to(device)
                prediction = lstm2(train_feature)
                train_loss = loss(train_label, prediction)
                running_loss += train_loss.item()
                optim.zero_grad()
                train_loss.backward()
                optim.step()

            train_total_loss.append(running_loss)
            writer.add_scalar('train_loss_'+file, running_loss, epoch + 1)
            # validate model, and save the best model
            lstm2.eval()
            validaton_loss = 0
            for data in val_loader:
                val_feature, val_label = data
                val_feature = val_feature.to(device)
                val_label = val_label.to(device)
                prediction = lstm2(val_feature)
                val_loss = loss(val_label, prediction)
                validaton_loss += val_loss.item()

            writer.add_scalar('val_loss_'+file, validaton_loss, epoch+1)
            val_total_loss.append(validaton_loss)
            # 把val_total_loss最小的MyLSTM2模型保存起来
            if validaton_loss < best_val_loss:
                best_epoch = epoch
                best_val_loss = validaton_loss
                torch.save(lstm2.state_dict(), './lstm2_best_models_final/lstm2_best_val_'+ file + '.pth')

            if epoch % 100 == 0:
                print(epoch, datetime.now(), running_loss, validaton_loss, best_val_loss, best_epoch)

        # 测试最好的模型，算出RMSE和MAPE
        # 重新定义MyLSTM2（用之前储存的val_total_loss最小的MyLSTM2模型的参数）
        best_lstm2 = MyLSTM2(6, 64, 6, 1).to(device)
        best_lstm2.load_state_dict(torch.load('./lstm2_best_models_final/lstm2_best_val_'+ file + '.pth'))

        # testing set
        test_feeder = Feeder(data_path, train_val_test='test')
        test_loader = torch.utils.data.DataLoader(dataset=test_feeder, batch_size=1, shuffle=False)

        test_total_SE = []
        test_total_APE = []
        for data in test_loader:
            test_feature, test_label = data
            test_feature = test_feature.to(device)
            test_label = test_label.to(device)
            prediction = best_lstm2(test_feature)
            test_SE = (test_label - prediction)**2
            test_APE = torch.abs(test_label - prediction) / test_label * 100
            test_total_SE.append(test_SE.item())
            test_total_APE.append(test_APE.item())
        test_RMSE_loss = np.sqrt(np.mean(test_total_SE))
        test_MAPE_loss = np.mean(test_total_APE)
        print(file + "_epoch是：{}".format(best_epoch))
        print(file + "_LSTM2模型的RMSE误差是：{}".format(test_RMSE_loss))
        print(file + "_LSTM2模型的MAPE误差是：{}".format(test_MAPE_loss))

    writer.close()



