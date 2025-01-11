import os
import random

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.metrics import r2_score


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

def bs_model(test_features):
    S = test_features[0]          # spot price
    X = test_features[2]          # strike
    dte = test_features[1] / 252  # days to maturity
    r = test_features[5] / 100    # interest rate
    sigma = test_features[4] / 100 # volatility
    d1 = (np.log(S / X) + (r + (sigma ** 2) / 2) * dte) / (sigma * (dte ** 0.5))
    d2 = d1 - sigma * (dte ** 0.5)
    C = S * norm.cdf(d1) - X * np.exp(-r * dte) * norm.cdf(d2)
    return C

results = []
root_path = './data/v1.1'
files = os.listdir(root_path)
files.remove(".DS_Store")
for file in files:
    print("Processing %s ..." % file)
    data_path = os.path.join(root_path, file)
    data = pd.read_csv(data_path)
    data = data[data['call price'] != 0]
    features = data.iloc[:, [1, 2, 3, 5, 6, 7]]
    features = features.to_numpy()
    features = features.astype(np.float32)
    labels = data.iloc[:, [4]]
    labels = labels.to_numpy()
    labels = labels.astype(np.float32)
    total_num = features.shape[0]
    # 打乱索引的顺序，然后features和labels按照索引取值
    permutation = np.random.permutation(total_num)
    features, labels = features[permutation], labels[permutation]
    test_idx_list = list(np.arange(int(total_num * 0.8), total_num))
    test_features = features[test_idx_list]
    test_labels = labels[test_idx_list]
    test_total_SE = []
    test_total_APE = []
    prediction = []
    for i in range(test_features.shape[0]):
        x = test_features[i]
        y_hat = bs_model(x)
        y = test_labels[i]
        test_SE = (y - y_hat) ** 2
        test_total_SE.append(test_SE)
        test_APE = abs(y - y_hat) / y * 100
        test_total_APE.append(test_APE)
        prediction.append(y_hat)
    test_RMSE = np.sqrt(np.mean(test_total_SE))
    test_MAPE = np.mean(test_total_APE)
    test_r_squared = r2_score(test_labels, prediction)
    print(file + "BS模型的RMSE误差是：{}".format(test_RMSE))
    print(file + "BS模型的MAPE误差是：{}".format(test_MAPE))
    print(file + "BS模型的R-squared是：{}".format(test_r_squared))
    results.append([test_RMSE, test_MAPE, test_r_squared])

df = pd.DataFrame(results, columns=['RMSE', 'MAPE', 'R-squared',],
                      index=files)

# 保存为Excel文件
df.to_excel('./bs_test_result.xlsx')







