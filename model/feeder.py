import torch.utils.data
import pandas as pd
import numpy as np


# 把初始的表格变成dataloader里面的参数dataset（分成训练，验证和测试集三类）
class Feeder(torch.utils.data.Dataset):
    def __init__(self, data_path, train_val_test='train'):
        data = pd.read_csv(data_path)
        data = data[data['call price'] != 0]
        features = data.iloc[:, [1, 2, 3, 5, 6, 7]]
        features = features.to_numpy()  # 将 DataFrame 转换为 NumPy 数组再进行归一化，因为许多归一化函数或库（如 Scikit-learn 中的 MinMaxScaler）接受的输入类型是 NumPy 数组，而不是 DataFrame。当然了归一化的对象必须是数值。
        # normalization
        f_min = np.min(features, axis=0)  # 计算features每列的最小值
        f_max = np.max(features, axis=0)
        features = (features - f_min) / (f_max - f_min)
        labels = data.iloc[:, [4]]
        labels = labels.to_numpy()
        # 划分训练，验证和测试集
        total_num = features.shape[0]
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
        elif train_val_test.lower() == 'all':
            self.features = features
            self.labels = labels
        else:
            raise Exception("参数train_val_test定义错误！")   # 如果参数不是以上三个数据集就报错

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


if __name__=="__main__":
    feeder = Feeder('../data/v1.1/op_money_0.97dte_9.csv', train_val_test='all')
    print(len(feeder))
    num = 0
    for i in range(len(feeder)):
        f, l = feeder[i]
        if l == 0.0:
            num += 1
            print(i, f, l)
    print(num)
    print(len(feeder))
    # loader = torch.utils.data.DataLoader(dataset=feeder, batch_size=30, shuffle=True)
    # for data in loader:
    #     features, labels = data
    #     print(data)



