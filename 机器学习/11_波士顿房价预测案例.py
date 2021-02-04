# 根据13个特征预测房屋价格中位数

import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.tree as st
import sklearn.ensemble as se
import sklearn.metrics as sm
import matplotlib.pyplot as mp
import numpy as np

# 读取数据
boston = sd.load_boston()

random_seed = 7  # 随机种子，用来产生随机值
x, y = su.shuffle(boston.data,  # 13个特征
                  boston.target,  # 标签，房屋价格中位数
                  random_state=7)  # 打乱样本
train_size = int(len(x) * 0.8)  # 计算训练集大小
# 划分训练集和测试集
train_x = x[:train_size]  # 训练集输入
train_y = y[:train_size]  # 训练集输出
test_x = x[train_size:]  # 测试集输入
test_y = y[train_size:]  # 测试集输出

# 定义模型
model = st.DecisionTreeRegressor(max_depth=4)  # 树的最大深度为4
model.fit(train_x, train_y)  # 训练
pred_test_y = model.predict(test_x)  # 使用测试集进行预测
# 计算并打印R2指标
print(sm.r2_score(test_y, pred_test_y))

# 特征重要性
fi = model.feature_importances_
print(fi)

# 特征重要性可视化
mp.figure("Feature importances", facecolor="lightgray")
mp.plot()
mp.title("DT Feature", fontsize=16)
mp.ylabel("Feature importances", fontsize=14)
mp.grid(linestyle=":")
x = np.arange(fi.size)
sorted_idx = fi.argsort()[::-1]  # 重要性排序(倒序)
fi = fi[sorted_idx]  # 根据排序索引重新排特征值
mp.xticks(x, boston.feature_names[sorted_idx])
mp.bar(x, fi, 0.4, color="dodgerblue", label="DT Feature importances")

mp.legend()
mp.tight_layout()
mp.show()