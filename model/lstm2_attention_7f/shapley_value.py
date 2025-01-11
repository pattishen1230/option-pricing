# 把训练集，测试集都保存成dataframe（可以从训练集和测试集中抽一部分来做，不一定全部）
import os
import numpy as np
import pandas as pd
import shap
import torch
from matplotlib import pyplot as plt
from train_val_test_lstm2_attention import set_random_seed
from model import MyLSTM2_ATN


set_random_seed()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(">>>>>>>>>>>> Training on ", device)
root_path = os.path.dirname(os.path.dirname(os.getcwd())) + '/data/v1.3'
files = os.listdir(root_path)
print(files)
for file in files:
    if file == ".DS_Store" or file != "op_money_1.03dte_180.csv":
        continue
    print("Processing %s ..." % file)
    data_path = os.path.join(root_path, file)
    data = pd.read_csv(data_path)
    features = data.iloc[:, [i for i in range(21)]]
    # normalization(使所有特征对最后结果影响的权重相同，避免梯度消失或爆炸）
    f_min = np.min(features, axis=0)  # 计算features每列的最小值
    f_max = np.max(features, axis=0)
    features = (features - f_min) / (f_max - f_min)
    features['call price1'] = 0
    # 把features当前的数据和最远的历史数据调换位置
    columns = features.columns
    new_order = list(columns[14:]) + list(columns[7:14]) + list(columns[:7])
    features = features[new_order]
    labels = data.iloc[:, [21]]

    total_num = features.shape[0]
    # 打乱索引的顺序，然后features和labels按照索引取值
    permutation = np.random.permutation(total_num)
    features, labels = features.iloc[permutation], labels.iloc[permutation]
    train_idx_list = list(np.arange(0, int(total_num * 0.7)))  # 获取训练集的索引的数字list（0，总行数的0.7倍）
    val_idx_list = list(np.arange(int(total_num * 0.7), int(total_num * 0.8)))
    test_idx_list = list(np.arange(int(total_num * 0.8), total_num))
    train_features = features.iloc[train_idx_list]
    train_labels = labels.iloc[train_idx_list]
    test_features = features.iloc[test_idx_list]
    test_labels = labels.iloc[test_idx_list]

    # 加载训练好的模型
    best_lstm2_atn = MyLSTM2_ATN(7, 64, 6, 1).to(device)
    best_lstm2_atn.load_state_dict(torch.load('./gru_attention_7f_best_models_final/lstm2_attention_best_val_' + file + '.pth', map_location=device))
    best_lstm2_atn.eval()
    def model_callable(x):
        x = x.to_numpy()  # (N, 21)
        x = x.astype(np.float32)
        x = x.reshape(x.shape[0], 3, 7)
        x = torch.tensor(x, device=device)
        output = best_lstm2_atn(x)
        return output.detach().cpu().numpy()

    # 初始化 SHAP 可视化工具
    shap.initjs()
    # 对不出席的feature进行随机采样（在训练集里面抽样，与训练集有相同的特征分布和形状）
    background_data = shap.maskers.Independent(train_features, max_samples=1)
    # y_train = background_data.data
    # y_train = pd.DataFrame(y_train)
    # y_train = model_callable(y_train)

    explainer = shap.Explainer(model_callable, background_data)  # 训练解释器时用训练集
    shap_values = explainer(test_features)  # 计算测试集样本的shap值
    # shap_values的形状：.values--所有shap value的矩阵；.base_values--模型在所有测试集上输出的prediction的均值，都一样；.data--训练集的x值
    # 验证shap算法可加性
    test_predictions = model_callable(test_features)
    #
    # print(test_predictions)
    # df = pd.DataFrame(shap_values.values)
    # pd.set_option("display.max_rows", None)  # 显示所有行
    # pd.set_option("display.max_columns", None)  # 显示所有列
    # print(df)

    print('预测值总和为：', np.sum(test_predictions))
    print('SHAP基准值总和为：', np.sum(shap_values.base_values))
    print('Shap Value总和为：', shap_values.values.sum())



    #画图（最后出来的是所有feature shap值的绝对值的均值 从大到小排列）
    plt.figure(dpi=400, figsize=(20,10)) # 设置图片的清晰度
    shap.summary_plot(shap_values.values, test_features, max_display=21, show=False, alpha=0.6)
    plt.tight_layout()
    plt.savefig('./shap_plot/final/'+file+'.png')  # 保存图片
    plt.close()
    # 全局的解释（把所有测试集数据都输入）
    shap.plots.bar(shap_values, max_display=21, show=False)  # 所有feature的绝对值的均值(显示所有的特征）
    plt.tight_layout()
    plt.savefig('./shap_plot/absolute/'+file+'.png')
    plt.close()
    shap.plots.bar(shap_values.abs.max(0), max_display=21, show=False) # 求绝对值的最大值
    plt.tight_layout()
    plt.savefig('./shap_plot/abs_max/' + file + '.png')
    # shap.dependence_plot(4, shap_values.values, np.array(test_features), feature_names=test_features.columns, show=False)
    # plt.tight_layout()
    # plt.savefig('./shap_plot/dependence_plot/' + file + '.png')

    # 局部的解释（输入某一个样本的shap值）
    # shap.plots.bar(shap_values[0])   # 默认不取绝对值
    # shap.plots.bar(shap_values[0].abs)   # 对自己取绝对值


