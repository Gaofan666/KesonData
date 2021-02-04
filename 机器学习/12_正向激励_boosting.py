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

# 定义模型（集成模型） 利用多个决策树进行计算，计算量变大
model = se.AdaBoostRegressor(  # 正向激励回归器
    st.DecisionTreeRegressor(max_depth=4),  # 基本模型：决策树
    n_estimators=400,  # 决策树数量
    random_state=random_seed  # 随机种子
)

model.fit(train_x, train_y)  # 训练
pred_test_y = model.predict(test_x)  # 预测
r2 = sm.r2_score(test_y, pred_test_y)
print('r2:', r2)  # 由0.8提升到了0.90
