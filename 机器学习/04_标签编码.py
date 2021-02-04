# 将字符串数据转换为数值，模型计算更加方便
import numpy as np
import sklearn.preprocessing as sp

raw_samples=np.array(['audi','fute','audi','bmw','ford','bmw'])

lbl_encoder = sp.LabelEncoder() # 创建标签编码
lbl_samples = lbl_encoder.fit_transform(raw_samples)

print(lbl_samples)

lbl_samples = lbl_encoder.inverse_transform(lbl_samples) # 逆向转换
print(lbl_samples)