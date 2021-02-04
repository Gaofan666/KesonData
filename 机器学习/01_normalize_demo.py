# 归一化示例

import numpy as np
import sklearn.preprocessing as sp

# 样本数据
raw_samples = np.array([[10.0, 20.0, 5],
                        [8.0, 10.0, 1.0]])
nor_samples = raw_samples.copy()  # 复制样本

for row in nor_samples:
    row /= abs(row).sum()  # 数字除以该行绝对值之和
print(nor_samples)

# 利用sklearn提供的API实现
nor_samples = sp.normalize(raw_samples,  # 原数组
                           norm='l1')  # 正则化方法 L1范数
print(nor_samples)

#  L1 : 除以向量中各元素绝对值之和  L2: 除以向量中各元素绝对值平方和的根
