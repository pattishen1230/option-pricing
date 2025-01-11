import os
import time
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from model import CNN1D

class Feeder(torch.utils.data.Dataset):
    def __init__(self, data_path, train_val_test='train'):
        data = pd.read_csv(data_path)
        data = data[data['call price'] != 0]
        features = data.iloc[:, [1, 2, 3, 5, 6, 7]]
        features = features.to_numpy()  # 将 DataFrame 转换为 NumPy 数组再进行归一化，因为许多归一化函数或库（如 Scikit-learn 中的 MinMaxScaler）接受的输入类型是 NumPy 数组，而不是 DataFrame。当然了归一化的对象必须是数值。
        features = features.astype(np.float32)
        # normalization
        f_min = np.min(features, axis=0)  # 计算features每列的最小值
        f_max = np.max(features, axis=0)
        features = (features - f_min) / (f_max - f_min)
        labels = data.iloc[:, [4]]
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
    writer = SummaryWriter('./logs_cnn1d')
    print(os.path.dirname(os.path.dirname(os.getcwd())))
    root_path = os.path.dirname(os.path.dirname(os.getcwd())) + '/data/v1.1'
    files = os.listdir(root_path)
    print(files)
    for file in files:
        if file == ".DS_Store":
            continue
        print("Processing %s ..." % file)
        data_path = os.path.join(root_path, file)
        train_feeder = Feeder(data_path, train_val_test='train')
        train_loader = torch.utils.data.DataLoader(dataset=train_feeder, batch_size=1024, shuffle=True)
        val_feeder = Feeder(data_path, train_val_test='val')
        val_loader = torch.utils.data.DataLoader(dataset=val_feeder, batch_size=1024, shuffle=True)
        print("Length of training set: %d ..." % len(train_feeder))
        # define a CNN1D
        cnn1d = CNN1D(1, 1, 3).to(device)
        optim = Adam(cnn1d.parameters(), lr=0.0001)

        train_total_loss = []
        val_total_loss = []
        loss = nn.MSELoss()
        best_val_loss = np.float32('inf')
        best_epoch = 0
        for epoch in range(2000):
            running_loss = 0
            cnn1d.train()
            for data in train_loader:
                train_feature, train_label = data
                train_feature = train_feature.to(device)
                train_feature = train_feature.unsqueeze(1)
                train_label = train_label.to(device)
                prediction = cnn1d(train_feature)
                train_loss = loss(train_label, prediction)
                running_loss += train_loss.item()
                optim.zero_grad()
                train_loss.backward()
                optim.step()

            train_total_loss.append(running_loss)
            writer.add_scalar('train_loss_'+file, running_loss, epoch + 1)
            # validate model, and save the best model
            cnn1d.eval()
            validaton_loss = 0
            for data in val_loader:
                val_feature, val_label = data
                val_feature = val_feature.to(device)
                val_feature = val_feature.unsqueeze(1)
                val_label = val_label.to(device)
                prediction = cnn1d(val_feature)
                val_loss = loss(val_label, prediction)
                validaton_loss += val_loss.item()

            writer.add_scalar('val_loss_'+file, validaton_loss, epoch+1)
            val_total_loss.append(validaton_loss)
            # 把val_total_loss最小的CNN1D模型保存起来
            if validaton_loss < best_val_loss:
                best_epoch = epoch
                best_val_loss = validaton_loss
                torch.save(cnn1d.state_dict(), './cnn1d_best_models_final/cnn1d_best_val_'+ file + '.pth')

            if epoch % 10 == 0:
                print(epoch, datetime.now(), running_loss, validaton_loss, best_epoch)

        # 测试最好的模型，算出RMSE和MAPE
        # 重新定义CNN1D（用之前储存的val_total_loss最小的CNN1D模型的参数）
        best_cnn1d = CNN1D(1, 1, 3)
        best_cnn1d.load_state_dict(torch.load('./cnn1d_best_models_final/cnn1d_best_val_'+ file + '.pth'))


        # testing set
        test_feeder = Feeder(data_path, train_val_test='test')
        test_loader = torch.utils.data.DataLoader(dataset=test_feeder, batch_size=1, shuffle=False)

        test_total_SE = []
        test_total_APE = []
        for data in test_loader:
            test_feature, test_label = data
            test_feature = test_feature.unsqueeze(1)
            prediction = best_cnn1d(test_feature)
            test_SE = (test_label - prediction)**2
            test_APE = torch.abs(test_label - prediction) / test_label * 100
            test_total_SE.append(test_SE.item())
            test_total_APE.append(test_APE.item())
        test_RMSE_loss = np.sqrt(np.mean(test_total_SE))
        test_MAPE_loss = np.mean(test_total_APE)
        print(file + "_epoch是：{}".format(best_epoch))
        print(file + "_CNN1D模型的RMSE误差是：{}".format(test_RMSE_loss))
        print(file + "_CNN1D模型的MAPE误差是：{}".format(test_MAPE_loss))

    writer.close()



