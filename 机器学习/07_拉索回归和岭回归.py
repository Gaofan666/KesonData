# lasso回归：在标准线性回归损失函数上添加L1范数
# 岭回归Ridge: 在标准线性回归损失函数的基础上添加L2范数
# 目的：让线性回归模型更趋向于大多数正常样本，给少量异常样本较轻的权重


import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as sm
import matplotlib.pyplot as mp
import sklearn.preprocessing as sp

# 读取数据
x, y = [], []  # 样本输入，输出
with open('./data/abnormal.txt', 'rt') as f:
    for line in f.readlines():
        # 将样本文件中的字符串拆分并转换为浮点数
        data = [float(substr) for substr in line.split(',')]
        x.append(data[:-1])
        y.append(data[-1])

# 转换为数组
x = np.array(x)
y = np.array(y)

# 标准线性回归
model = lm.LinearRegression()
model.fit(x, y)  # 训练
pred_y = model.predict(x)  # 用训练集预测

# 岭回归
model_2 = lm.Ridge(alpha=200, max_iter=1000)  # alpha越大，偏离奇异值越远
model_2.fit(x, y)
pred_y2 = model_2.predict(x)

# 拉索回归
model_3 = lm.Lasso(alpha=0.5, max_iter=1000)  # alpha同上
model_3.fit(x, y)
pred_y3 = model_3.predict(x)

# 可视化回归曲线
mp.figure('Linear & Ridge & Lasso', facecolor='lightgray')
mp.title('Linear & Ridge & Lasso', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.scatter(x, y, c='dodgerblue', alpha=0.8, s=60, label='Sample')
sorted_idx = x.T[0].argsort()  # x是所有的样本（50，1）
# x.T是(1,50)   .argsort是从小到大排序  下面pred_y 也是从小到大排序
mp.plot(x[sorted_idx], pred_y[sorted_idx], c='orangered', label='Linear')  # 线性回归
mp.plot(x[sorted_idx], pred_y2[sorted_idx], c='limegreen', label='Ridge')  # 岭回归
mp.plot(x[sorted_idx], pred_y3[sorted_idx], c='blue', label='Lasso')  # Lasso回归

mp.legend()
mp.show()
