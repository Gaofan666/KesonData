# 模型保存示例
import numpy as np
import sklearn.linear_model as lm  # 线性模型
import pickle

x = np.array([[0.5], [0.6], [0.8], [1.1], [1.4]])  # 输入集
y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])  # 输出集

# 创建线性回归器
model = lm.LinearRegression()
# 用已知输入，输出数据集训练回归器
model.fit(x, y)
print('训练完成')

# 打开文件（二进制写入模式）
with open('Linear_model.pkl', 'wb') as f:
    pickle.dump(model, f)  # 保存
    print('保存模型成功！')
