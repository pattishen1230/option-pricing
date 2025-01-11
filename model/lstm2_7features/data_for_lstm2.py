import datetime
import os
from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm import tqdm

print(os.path.dirname(os.path.dirname(os.getcwd())))
root_path = os.path.dirname(os.path.dirname(os.getcwd())) + '/data/v1.1'
files = os.listdir(root_path)
if ".DS_Store" in files:
    files.remove('.DS_Store')
print(files)

for file in files:

    print("Processing %s ..." % file, datetime.datetime.now())
    data_path = os.path.join(root_path, file)
    data = pd.read_csv(data_path)
    data = data.iloc[:, [0, 1, 2, 3, 5, 6, 7, 4]]
    data = data[data['call price'] != 0]
    data = data.reset_index(drop=True)
    data['date'] = pd.to_datetime(data['date'].astype(str), format="%Y%m%d")
    first_date = data.iloc[0, 0]
    grouped_data = data.groupby('date')
    data_dict = {date: group.drop('date', axis=1).to_numpy().astype(np.float32) for date, group in grouped_data}

    sequence_length = 3
    new_features = []
    new_labels = []

    pbar = tqdm(total=len(data_dict))
    for current_date, current_features_labels in data_dict.items():
        pbar.update(1)
        for i in range(current_features_labels.shape[0]):
            current_strike = current_features_labels[i, 2]
            current_dte = int(current_features_labels[i, 1])
            # 初始化一个列表，用于存储当前组数据的三行
            current_features_group = [deepcopy(current_features_labels[i, :])]  # deepcopy不会改变原current_features_labels
            # current_features_group[0][-1] = 0

            for j in range(len(data)):
                # 获取相应的日期
                target_date = current_date - pd.Timedelta(days=j + 1)
                if target_date < first_date:
                    break
                target_dte = current_dte + (j + 1)

                if target_date in data_dict:
                    target_features_labels = data_dict[target_date]
                    # 找到相应日期符合条件的数据行的索引
                    target_index = np.where((target_features_labels[:, 1] == target_dte) & (target_features_labels[:, 2] == current_strike))[0]
                    # 找到了符合条件的数据行，将其添加到当前数据组中
                    if len(target_index) > 0:
                        current_features_group.append(target_features_labels[target_index[0], :])

            # 如果当前数据组中的数据行数量等于 sequence_length，说明找到了完整的一组数据（可以供LSTM2使用的数据组）
            if len(current_features_group) == sequence_length:
                # 将三行数据组成的新数据添加到新数据集中
                new_features.append(current_features_group)
                # 将相应的 "call price" 标签添加到新标签集中
                new_labels.append(current_features_labels[i, -1])
    pbar.close()

    # 转换为 NumPy 数组
    new_features = np.array(new_features)
    new_labels = np.array(new_labels)
    print(new_features.shape)
    print(new_labels.shape)
    new_features = new_features.reshape(len(new_features), 21)
    new_labels = new_labels.reshape(len(new_labels), 1)
    data_list = np.concatenate((new_features, new_labels), axis=1)
    df =pd.DataFrame(data=data_list, columns=['spot price1', 'days to maturity1', 'strike1', 'moneyness1', 'volatility1', 'interest rate1', 'call price1',
                                       'spot price2', 'days to maturity2', 'strike2', 'moneyness2', 'volatility2', 'interest rate2', 'call price2',
                                       'spot price3', 'days to maturity3', 'strike3', 'moneyness3', 'volatility3', 'interest rate3', 'call price3',
                                       'call price'])

    # 将 DataFrame 保存为 CSV 文件
    df.to_csv(os.path.dirname(os.path.dirname(os.getcwd())) + '/data/v1.3/' + file, index=False)
