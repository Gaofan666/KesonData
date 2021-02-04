"""
5）编写代码，实现以下功能：

- 给出10个x值，计算y， y=2x

- 在计算出的y值上加入噪声值（噪声值范围1~3之间）

- 使用第一步、第二步产生的x,y作为样本数据，执行线性回归

- 可视化原始样本、预测结果
"""

import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as sm
import matplotlib.pyplot as mp
import random

x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = 2.0 * x

# 利用随机函数产生噪声 默认噪声在0-1之间
random_num = np.random.rand(len(y), 1) * 3  # 生成y的长度个噪声
print(random_num)
y = y + random_num  # 在y上加上噪声数据

# 定义线性模型
model = lm.LinearRegression()
model.fit(x, y)
pred_y = model.predict(x)

print('coef_:', model.coef_)  # 系数
print('intercept_:', model.intercept_)  # 截距

# 可视化
mp.scatter(x, y, marker='D', c='blue', label='Samples')
mp.plot(x, pred_y, c='red', label='Linear')

mp.legend()
mp.show()
