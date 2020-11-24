# DataFrame示例  一行一样本，一列一特征

import numpy as np
import pandas as pd

df = pd.DataFrame()
print(df)

print('-' * 15, '通过列表创建DataFrame', '-' * 15)
ary = np.array([1, 2, 3, 4, 5])
df = pd.DataFrame(ary)
print(df)
print('维度：', df.shape)

print('-' * 15, '更改索引名称', '-' * 15)
data = [['Alex', 10], ['Bob', 12], ['Clarke', 13]]
df = pd.DataFrame(data, index=['s1', 's2', 's3'], columns=['Name', 'Age'])
print(df)

print('-' * 15, '通过字典创建DataFrame', '-' * 15)
# 字典里面的key代表一个特征(列级索引)，values是各种特征值  可以理解为一个字典就是一列
data = {'Name': ['Tom', 'Jack', 'Steve', 'Ricky'], 'Age': [28, 34, 29, 42]}
df = pd.DataFrame(data, index=['s1', 's2', 's3', 's4'])  # 别忘了更改行级索引
print(df)

print('-' * 15, '记住这四个方法', '-' * 15)
print('返回列标签', df.columns)
print('返回行标签', df.index)
print('返回前两行')
print(df.head(2))
print('返回后两行')
print(df.tail(2))

print('-' * 15, 'Age少了一个值，如果是列表会报错，但是Series不会报错，填充NaN', '-' * 15)
data = {'Name': pd.Series(['Tom', 'Jack', 'Steve', 'Ricky']), 'Age': pd.Series([28, 34, 29])}
df = pd.DataFrame(data)  # 更改行索引后全是NaN ???
print(df)

d = {'one': pd.Series([1, 2, 3], index=['a', 'b', 'c']),
     'two': pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd']),
     'three': pd.Series([10, 20, 30], index=['a', 'b', 'c'])}

print('-' * 15, '列的访问', '-' * 15)
df = pd.DataFrame(d)
print(df)
print(df['two'])  # Series
print(df[['two']])  # DataFrame 加了一个中括号
print(df[['one', 'two']])  # 要加两个中括号将one two定为一个整体
print(df[df.columns[:-1]])  # 用切片取前两列

print('-' * 15, '列的添加', '-' * 15)
df['four'] = pd.Series([90, 50, 60, 40], index=['a', 'b', 'd', 'c'])
print(df)

print('-' * 15, '列的删除', '-' * 15)
# 返回删除结果但原df不变 1水平 0垂直
print(df.drop(['three', 'four'], axis=1))  # drop可以删除好几行

print('-' * 15, '行的访问', '-' * 15)
print(df)
print(df.loc['b'])  # 拿到b的一行 返回Series
print(df.loc[['b', 'c']])  # 拿到b,c两行
print(df.loc['b':'c'])  # 切片
# 用iloc  -->  index loc
print(df.iloc[1])
print(df.iloc[[1, 2]])  # 取1，2行
print(df.iloc[:-1])  # 取前三行

print('-' * 15, '行的添加', '-' * 15)
print(df)
# 添加一行,别忘了更改列名，不然整合的时候索引无法对应
newdf = pd.DataFrame([[10, 20, 80, 30]], columns=df.columns, index=['e'])
df = df.append(newdf)
print(df)

print('-' * 15, '行的删除', '-' * 15)
# drop方法  axis=0删行  =1删列
print(df.drop(['c', 'd'], axis=0))  # 删除cd这两行

# 读取电信用户数据
# 把pack_type,extra_flow,loss存入DataFrame,获取前5行
# 加载数据
with open('CustomerSurvival.csv', 'r') as f:
    data = []
    # index 行数索引(0，1，2...)  line 每一行的数据（包括了i）[i,数据]
    for index, line in enumerate(f.readlines()):
        # print(line)
        # print(line[:-1])
        # print(line[:-1].split(','))
        # 将每一行的数据按照逗号拆分并且存到元组当中，放到Data当中
        row = tuple(line[:-1].split(','))  # line[:-1]是数据
        data.append(row)

    # 转成ndarry
    # pack_type 套餐金额（1.<96 2.96-225 3.>225）
    # pack_change 改变行为(是否曾经改变过套餐金额 1.是 0.否)
    # contract 服务合约：用户是否与联通签订过服务合约 1.是 0.否
    # asso_pur 关联购买：用户再使用联通移动服务过程中是否同时办理其他业务
    # （主要是固定电话和宽带业务 1.同时办理一项其他业务 2.同时办理两项其他业务）
    # group_user 集团用户(联通内部人员)
    data = np.array(data, dtype={
        'names': ['index', 'pack_type', 'extra_time', 'extra_flow', 'pack_change',
                  'contract', 'asso_pur', 'group_user', 'use_month', 'loss'],
        'formats': ['i4', 'i4', 'f8', 'f8', 'i4', 'i4', 'i4', 'i4', 'i4', 'i4']
    })

# 瘦身： 只需要 pack_type  extra_time  loss
sub_df = pd.DataFrame({'pack_type': data['pack_type'],
                       'extra_time': data['extra_time'],
                       'loss': data['loss']})
print(sub_df)
# 追加一列 extra_flow
sub_df['extra_flow'] = data['extra_flow']
print(sub_df.head(10))
# 选择所有未流失数据行
unloss_df = sub_df.loc[sub_df['loss'] == 0]  # 这里加不加loc都对
print(unloss_df.head(3))

print('-' * 15, '修改DataFrame的某个数据', '-' * 15)
print(df)
# 先找列再找行 four列 a行
df['three']['d'] = 66
print(df)

print('-' * 15, '复合索引', '-' * 15)
