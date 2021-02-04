import numpy as np
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import axes3d
import sklearn.preprocessing as sp

# 训练数据集
train_x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])  # 输入集
train_y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])  # 输出集

lrate = 0.01  # 学习率 0-1
epochs = []  # 记录迭代次数
losses = []  # 记录损失函数的值
w0, w1 = [1], [1]  # 线性模型的两个参数，初始值都设置为1
for i in range(1, 1001):
    # 取最新的w0w1执行计算（预测）
    y = w0[-1] + w1[-1] * train_x  # 计算 y=w0 + w1*x
    # 计算损失值
    loss = (((train_y - y) ** 2).sum()) / 2  # 均方差
    print('%d:w0=%f,w1=%f,loss=%f' % (i, w0[-1], w1[-1], loss))
    # 计算d0 d1 (参数更新大小)
    d0 = -(train_y - y).sum()
    d1 = -((train_y - y) * train_x).sum()
    w0.append(w0[-1] - (d0 * lrate))
    w1.append(w1[-1] - (d1 * lrate))
