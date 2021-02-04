import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array([[1, 3, 2],
                        [7, 5, 4],
                        [1, 8, 6],
                        [7, 3, 9]])

# 定义独热编码器
one_hot_encoder = sp.OneHotEncoder(
    sparse=False,  # 稀疏格式：否
    dtype='int32',  # 元素类型
    categories='auto'  # 自动编码
)

one_samples = one_hot_encoder.fit_transform(raw_samples)
print(one_samples)
