# 正向激励示例
import sklearn.datasets as sd
import sklearn.utils as su
import sklearn.tree as st
import sklearn.ensemble as se
import sklearn.metrics as sm

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
model = se.RandomForestRegressor(
    max_depth=10, # 最大深度
    n_estimators=1000, # 树的数量
    min_samples_split=2 # 最少样本数量
)

model.fit(train_x, train_y)  # 训练
pred_test_y = model.predict(test_x)  # 预测
r2 = sm.r2_score(test_y, pred_test_y)
print('r2:', r2)  # 由0.8提升到了0.92
