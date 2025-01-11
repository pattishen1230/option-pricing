import pandas as pd
import os

def first_process():
    '''
    :return: 从原始数据先提取出我需要的列，然后删去包含str类型的行
    对到期日进行了四舍五入并取整的操作 TODO 注意：这里可能会有问题，以后可以用两个日期相减重新计算
    接下来又计算了C_PRICE和moneyness
    最后保存为option_pricing.csv
    '''
    data = pd.read_csv('data/spy_2020_2022.csv')
    # 按照' [QUOTE_DATE]'列对整个dataframe进行排序
    data = data.sort_values(' [QUOTE_DATE]')
    data = data.iloc[:, [2, 4, 7, 17, 18, 19, ]]
    # 查看' [QUOTE_DATE]'列所有的数据类型并分别计数（全都是str）
    # data_types = df[' [QUOTE_DATE]'].apply(lambda x: type(x).__name__).value_counts()
    # 将日期列str转换为日期类型
    data[' [QUOTE_DATE]'] = pd.to_datetime(data[' [QUOTE_DATE]'])
    # 将日期列转换为数值类型的日期（如：20201023）
    data[' [QUOTE_DATE]'] = data[' [QUOTE_DATE]'].dt.strftime('%Y%m%d').astype(int)
    # 使用条件过滤删除包含字符串的行（其实是把所有不是字符串的行筛选出来，重新保存成data）（dataframe可以进行bool逻辑判断切片-True)
    data = data[~data.applymap(lambda x: isinstance(x, str)).any(axis=1)] # any(axis=1)看每一行的任何列是否包含str
    data['C_PRICE'] = (data[' [C_BID]'] + data[' [C_ASK]']) / 2
    data[' [DTE]'] = data[' [DTE]'].round().astype(int)  # 对到日期进行四舍五入
    data['Moneyness'] = data[' [UNDERLYING_LAST]']/data[' [STRIKE]']
    print(data.head())  # 默认显示数据的前五行
    data.to_csv('data/option_pricing.csv', index=False)  # 第二个参数：不显示索引


def second_process():
    '''
    :return: 把第一步处理好的数据按照moneyness（三个区间）和dte（五个区间）两个标准进行分类
    分类得到15个子表格分门别类的保存，以待后用。(存在processed文件夹里)
    分类逻辑：先对moneyness进行分类（for-loop），在第一个moneyness类别下，再对dte做分类，每一个moneyness大类下的dte分类得到的子类保存起来
    '''
    df = pd.read_csv('data/option_pricing.csv')
    money = [0, 0.97, 1.03, float('inf')]     # 包含所有区间端点值
    dte = [0, 9, 30, 90, 180, float('inf')]
    for i in range(len(money)-1):
        lower_b = money[i]
        upper_b = money[i+1]
        # 对'Moneyness'列所有元素进行符合上下界的判断，把返回bool值为true的行存成一个新的df_m（bool索引）
        df_m = df[(df['Moneyness'] <= upper_b) & (df['Moneyness'] > lower_b)]
        for j in range(len(dte)-1):
            lower_b = dte[j]
            upper_b = dte[j+1]
            df_d = df_m[(df_m[' [DTE]'] <= upper_b) & (df_m[' [DTE]'] > lower_b)]
            df_d.to_csv('data/processed/op_money_'+ str(money[i+1]) + 'dte_' + str(dte[j+1]) + '.csv', index=False)


def v1_process(file_name='op_money_infdte_9.csv', vol_name='vix9d.csv', rate_name='rate9d.csv'):
    '''
    :param file_name: 不包含vol和rate的表格
    :param vol_name: 需要加的vol
    :param rate_name: 需要加的rate
    :return: 按照file_name中的时间加上vol_name和rate_name两列（把processed文件夹里的表格操作后存到v1.0)
    '''
    df1 = pd.read_csv('data/processed/' + file_name)  # 待加的数据

    df2 = pd.read_csv('data/volatility/' + vol_name)  # 包含vol的数据
    if vol_name == 'vix1y.csv':
        df2 = df2.iloc[:, [0, 1]]  # vix1y的数据格式跟其他的不一样，单独取需要的列出来
    else:
        df2 = df2.iloc[:, [0, 5]]
    # 把vol数据里的时间列变成数值（如：20201023）
    df2['Date'] = pd.to_datetime(df2['Date'])
    df2['Date'] = df2['Date'].dt.strftime('%Y%m%d').astype(int)
    # how='left'：左连接将保留左侧df1的所有行，并根据合并键（' [QUOTE_DATE]'和'Date'）将右侧df2中匹配的行合并到左侧。
    merged_df = pd.merge(df1, df2, left_on=' [QUOTE_DATE]', right_on='Date', how='left')
    # 删除vol表格里的Date列（，inplace=True删除好后覆盖原数据，而不返回一个新的DataFrame）
    # axis=1表示沿着1轴（行的方向）遍历，先选'Date'列，再遍历所有的行（一个个删除所有的行）--主要看选择删除的是行还是列
    merged_df.drop('Date', axis=1, inplace=True)
    # 删除包含缺失值的行，缺失值通常表示为 NaN（Not a Number）或其他形式的缺失值标记。
    merged_df.dropna(inplace=True)
    # 删除包含缺失值的列：merged_df.dropna(axis=1, inplace=True)

    # 对df3进行上述同样的操作，并把df3添加到前面处理好的merged_df里
    df3 = pd.read_csv('data/interest rate/' + rate_name)
    df3['DATE'] = pd.to_datetime(df3['DATE'])
    df3['DATE'] = df3['DATE'].dt.strftime('%Y%m%d').astype(int)
    merged_df = pd.merge(merged_df, df3, left_on=' [QUOTE_DATE]', right_on='DATE', how='left')
    merged_df.drop('DATE', axis=1, inplace=True)
    merged_df.dropna(inplace=True)

    merged_df.to_csv('data/v1.0/' + file_name, index=False)


def rename_column():
    '''
    :return: 把最终包含所有input的表格（v1.0)分别对每列进行重新命名并删除包含str的行，存在v1.1
    注意：命名后反复warning-mixed types，经对column type检查发现，interest rate的数据类型全部为str，于是把命名后最后rate列str类型改为数值，大功告成！！！
    '''
    files = os.listdir('data/v1.0')  # 使用Python内置函数来获取指定目录中的所有文件和文件夹的名称列表。
    for file in files:
        if file == ".DS_Store":      # macOS 操作系统在目录中自动生成的隐藏文件，用于存储文件夹的自定义属性和元数据。
            continue                 # 终止当前循环迭代，并跳过剩余代码，开始下一次循环迭代。（不对".DS_Store"文件进行任何处理）
        print("Processing %s ..." % file)
        # 读取表格
        df = pd.read_csv('data/v1.0/' + file)
        df.drop(' [C_BID]', axis=1, inplace=True)
        df.drop(' [C_ASK]', axis=1, inplace=True)

        # 定义列名映射字典
        # 字典是一种无序的数据结构，其中的每个元素都由一个键和一个值组成。每个键-值对在字典中是唯一的，可以通过键来访问和操作对应的值。
        column_mapping = {
            df.columns[0]: 'date',
            df.columns[1]: 'spot price',
            df.columns[2]: 'days to maturity',
            df.columns[3]: 'strike',
            df.columns[4]: 'call price',
            df.columns[5]: 'moneyness',
            df.columns[6]: 'volatility',
            df.columns[7]: 'interest rate',
        }
        # 使用 rename() 方法修改列名
        df.rename(columns=column_mapping, inplace=True)

        # 查看'interest rate'列所有的数据类型并分别计数（全都是str）
        # data_types = df['interest rate'].apply(lambda x: type(x).__name__).value_counts()
        # 将'interest rate'列转换为数值类型（将无法转换为数值的值设置为缺失值NaN）
        df['interest rate'] = pd.to_numeric(df['interest rate'], errors='coerce')
        # 使用条件过滤删除包含字符串的行（只留下包含float的行）
        df = df[df['volatility'].apply(lambda x: isinstance(x, float))]
        df.dropna(inplace=True)
        df.to_csv('data/v1.1/' + file, index=False)

def identify_type_for_column(file='data/interest rate/rate1m.csv', col_name='DTB4WK'):
    df = pd.read_csv(file)
    data_types = df[col_name].apply(lambda x: type(x).__name__).value_counts()
    print(data_types)


if __name__ == "__main__":
    # 这行代码的含义：直接run本脚本时此行代码下的内容会被运行，本脚本被当成模块导入其他脚本时此代码下的内容不会被运行
    # 对15个子表格都进行v1_process的操作，每次处理好的表格存到v1.0下面
    # v1_process('op_money_infdte_9.csv', 'vix9d.csv', 'rate9d.csv')
    # 对v1.0文件夹下面的所有表格都进行重命名，保存为v1.1(代码只要运行一次，里面用了for-loop）
    # rename_column()

    # 检查看命名好的v1.1下的表格还有没有warning-mixed type
    files = os.listdir('data/v1.1')
    for file in files:
        if file == ".DS_Store":
            continue
        df = pd.read_csv('data/v1.1/' + file)
        print(df)


