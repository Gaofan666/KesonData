import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as sm  # 性能评价
import matplotlib.pyplot as mp
import sklearn.pipeline as pl  # 管线模型
import sklearn.preprocessing as sp  # 预处理模块

# 读取样本
train_x, train_y = [], []
with open('./data/poly_sample.txt', 'rt') as f:
    for line in f.readlines():  # 遍历每一行
        # 将每一行通过逗号拆分后转换成浮点数
        data = [float(substr) for substr in line.split(',')]
        train_x.append(data[:-1])  # 切出1-倒数第一个字段 二维
        train_y.append(data[-1])  # 直接取出最后一个字段 一维

# 列表转成数组
train_x = np.array(train_x)
train_y = np.array(train_y)
print(train_x.shape)
print(train_y.shape)

# 定义模型
# 管线模型，用来连接两个模型  多项式模型的参数是线性模型
model = pl.make_pipeline(
    # 我看 最高次12 是拟合最好的，13就过拟合了
    sp.PolynomialFeatures(12),  # 多项式扩展，最高次3
    lm.LinearRegression()  # 线性模型
)
model.fit(train_x, train_y)  # 训练
pred_train_y = model.predict(train_x)  # 预测

# 打印R2评价指标
r2 = sm.r2_score(train_y, pred_train_y)
print('R2:', r2)

# 可视化回归曲线
test_x = np.linspace(train_x.min(), train_x.max(), 1000)
pre_test_y = model.predict(test_x.reshape(-1, 1))

mp.figure('Polynomial')
mp.title('Polynomial', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.xlabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.scatter(train_x, train_y,
           c='blue', alpha=0.8, s=60, label='Sample')
mp.plot(test_x, pre_test_y, c='orangered', label='Regression')
mp.legend()
mp.show()