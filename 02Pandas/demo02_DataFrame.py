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

df = pd.DataFrame({'pack_type': data['pack_type'],
                   'extra_time': data['extra_time'],
                   'loss': data['loss']})
print(df.head(10))
