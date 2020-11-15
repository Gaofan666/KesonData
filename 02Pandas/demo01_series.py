"""Series对象"""
import numpy as np
import pandas as pd

# 创建Series对象
ary = np.array(['zs', 'ls', 'ww', 'zl'])
s = pd.Series(ary)
print(s)
# 使用index参数可以更改索引
s = pd.Series(ary, index=['s01', 's02', 's03', 's04'])
print(s)
# 使用字典创建Series
s = pd.Series({'s01': 'zs', 's02': 'ls'})
print(s)
# 使用标量创建Series  index可以控制里面有多少个元素
s = pd.Series(5, index=[0, 1, 2, 3, 4])  # index=np.arange(5)
print(s)

# Sesries的访问
s = pd.Series(ary, index=['s01', 's02', 's03', 's04'])
print('-' * 50)
print(s[1], s['s02'])
print(s[1:3])  # 包头不包尾
print(s['s02':'s04'])  # 包头包尾
mask = [True, True, False, True]  # 掩码
print(s[mask])  # 掩码也可以用在Series
print(s[[0, 2]])
print(s[['s01', 's02']])

# 四个元素倒序输出
print(s[::-1])
print(s[[3, 2, 1, 0]])

print(s.index)
print(s.values)

# 测试日期类型数据
print('-'*50)
dates = pd.Series(['2011', '2011-02', '2011-03-01', '2011/04/01',
                   '2011/05/01 01:01:01', '01 Jun 2011'])
dates = pd.to_datetime(dates)
print(dates)

# 查看距离20110101多少纳秒ns
delta = dates - pd.to_datetime('20110101')
print(delta)