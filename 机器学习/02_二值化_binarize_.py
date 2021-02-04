# 根据设定的阈值，比较数值和阈值的关系，如果大于阈值，设定为1（通常）
#   小于则设置为0（通常）

import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array([[65.5, 89.0, 73.0],
                        [55.0, 99.0, 98.5],
                        [45.0, 22.5, 60.0]])

bin_samples = raw_samples.copy()  # 复制数组
# 对数组中的值进行判断
mask1 = bin_samples < 60  # <60的元素返回True
mask2 = bin_samples >= 60
bin_samples[mask1] = 0
bin_samples[mask2] = 1
print(bin_samples)
print('-'*20)

bin = sp.Binarizer(threshold=59)  # 创建二值化对象
bin_samples = bin.transform(raw_samples)  # 二值转换
print(bin_samples)
