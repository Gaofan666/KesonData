import numpy as np
import pandas as pd

# 加载文件,指定列名，指定索引列为index这一列
data = pd.read_csv('./CustomerSurvival.csv', header=None,
                   names=['index', 'pack_type', 'extra_time', 'extra_flow', 'pack_change',
                          'contract', 'asso_pur', 'group_user', 'use_month', 'loss'],
                   index_col='index',
                   usecols=['index','pack_type','extra_flow','loss'])

# data.reset_index() # 重置索引
print(data)
