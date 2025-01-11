# Cox-Ross-Rubinstein for American style options
# 此代码改自GitHub: https://github.com/xdw15/Cox-Ross-Rubinstein-for-American-style-options-/blob/master/CoxRossRubinstein.py
import os
import random
import numpy as np
import pandas as pd
import torch
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



def crr(test_features):
    # Assigning paramter values
    S = test_features[0]  # spot price
    K = test_features[2]  # strike
    steps = int(test_features[1])   # days to maturity
    r = test_features[5] / 100  # interest rate
    sigma = test_features[4] / 100  # volatility
    dt = 1 / 252
    u = np.exp(dt ** 0.5 * sigma)
    d = np.exp(-(dt ** 0.5 * sigma))
    p = (np.exp(r * dt) - d) / (u - d)

    # Pre-allocating the underlying's price array
    pricetree = np.zeros([steps + 1, steps + 1], dtype='float64')  # 创建了一个steps+1行steps+1列的二维数组，初始值都为0，数据类型为float64。
    pricetree[0, 0] = S                                            # 第0行第0列为spot price

    # The tree is calculated for the underlying's price，模拟stock price path
    for i in range(1, steps + 1):
        pricetree[:i, i] = pricetree[:i, i - 1] * u
        pricetree[i, i] = pricetree[i - 1, i - 1] * d

    # Tree for the option value
    optiontree = np.zeros([steps + 1, steps + 1], dtype='float64')
    optiontree[:, -1] = np.maximum(pricetree[:, -1] - K, 0)   # option value的最后一列是max(S-K, 0)
    for i in range(steps - 1, -1, -1):
        optiontree[:i + 1, i] = np.exp(-(r*dt))*(optiontree[:i + 1, i + 1] * p + (1-p) * optiontree[1:i + 2, i + 1])
        # Early exercise
        optiontree[:i + 1, i] = np.maximum(pricetree[:i + 1, i] - K, optiontree[:i + 1, i])

    return optiontree[0, 0]

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
        y_hat = crr(x)
        y = test_labels[i]
        test_SE = (y - y_hat) ** 2
        test_total_SE.append(test_SE)
        test_APE = abs(y - y_hat) / y * 100
        test_total_APE.append(test_APE)
        prediction.append(y_hat)
    test_RMSE = np.sqrt(np.mean(test_total_SE))
    test_MAPE = np.mean(test_total_APE)
    test_r_squared = r2_score(test_labels, prediction)
    print(file + "CRR模型的RMSE误差是：{}".format(test_RMSE))
    print(file + "CRR模型的MAPE误差是：{}".format(test_MAPE))
    print(file + "CRR模型的R-squared是：{}".format(test_r_squared))
    results.append([test_RMSE, test_MAPE, test_r_squared])

df = pd.DataFrame(results, columns=['RMSE', 'MAPE', 'R-squared'], index=files)

# 保存为Excel文件
df.to_excel('./crr_test_result.xlsx')


