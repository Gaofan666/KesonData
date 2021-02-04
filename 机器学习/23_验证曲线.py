# 验证不同的参数对模型的影响

# 验证曲线示例
import numpy as np
import sklearn.preprocessing as sp
import sklearn.ensemble as se
import sklearn.model_selection as ms
import matplotlib.pyplot as mp

data = []
with open("./data/car.txt", "r") as f:
    for line in f.readlines():
        data.append(line.replace("\n", "").split(","))

data = np.array(data).T  # 转置
encoders, train_x = [], []

# 对样本数据进行标签编码
for row in range(len(data)):
    encoder = sp.LabelEncoder()  # 创建标签编码器
    encoders.append(encoder)
    if row < len(data) - 1:  # 不是最后一行，为样本特征
        lbl_code = encoder.fit_transform(data[row])  # 编码
        train_x.append(lbl_code)
    else:  # 最后一行，为样本输出
        train_y = encoder.fit_transform(data[row])

train_x = np.array(train_x).T  # 转置回来，变为编码后的矩阵

# 定义模型
model = se.RandomForestClassifier(max_depth=8,  # 最大深度
                                  random_state=7  # 随机种子
                                  )
# 产生数组，用于验证
n_estimators = np.arange(50, 550, 50)  # 产生一个数组
print('n_estimators:', n_estimators)

# 通过不同的参数，构建多个随机森林，验证其准确率
train_scores, test_scores = ms.validation_curve(
    model,
    train_x, train_y,
    'n_estimators',  # 待验证的参数名称
    n_estimators,  # 待验证的参数值
    cv=5  # 折叠数量
)
# print(test_scores)

train_mean = train_scores.mean(axis=1)  # 求各个折叠下性能均值
test_mean = test_scores.mean(axis=1)  # 求各个折叠下测试性能均值

# 可视化
mp.figure('n_estimators', facecolor='lightgray')
mp.title('n_estimators', fontsize=20)
mp.xlabel('n_estimators', fontsize=14)
mp.ylabel('F1 Score', fontsize=14)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.plot(n_estimators, test_mean, 'o-', c='blue', label='Testing')
mp.legend()
mp.show()