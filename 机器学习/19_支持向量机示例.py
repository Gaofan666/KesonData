"""
1.支持向量机是二分类模型
2.寻找最优的线性模型实现类别划分
3.支持向量到分类边界间隔最大化
4.两边支持向量到分类边界距离要相等
5.对于线性不可分问题，由核函数转换为线性可分
"""

import numpy as np
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.metrics as sm
import matplotlib.pyplot as mp

x, y = [], []
with open('./data/multiple2.txt', 'r') as f:
    for line in f.readlines():
        data = [float(substr) for substr in line.split(',')]
        x.append(data[:-1])  # 切出输入部分
        y.append(data[-1])  # 切出输出部分
# 转数组
x = np.array(x)
y = np.array(y, dtype=int)

# 定义模型
model = svm.SVC(kernel='linear')  # 线性核函数
model.fit(x,y) # 训练


