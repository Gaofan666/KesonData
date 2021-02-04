import numpy as np
import sklearn.model_selection as ms
import sklearn.preprocessing as sp
import sklearn.ensemble as se

raw_samples = []  # 保存一行样本数据
with open("./data/car.txt", "r") as f:
    for line in f.readlines():
        raw_samples.append(line.replace("\n", "").split(","))
data = np.array(raw_samples).T  # 转置
encoders = []  # 记录标签编码器
train_x = []  # 编码后的x
train_y = []

# 对样本数据进行标签编码
for row in range(len(data)):
    encoder = sp.LabelEncoder()  # 创建标签编码器
    encoders.append(encoder)
    if row < len(data) - 1:  # 不是最后一行，为样本特征
        lbl_code = encoder.fit_transform(data[row])  # 编码
        train_x.append(lbl_code)
    else:  # 最后一行，为样本输出
        train_y = encoder.fit_transform(data[row])

train_x = np.array(train_x).T  # 转置回来，变为编码后的数组
print(train_x)

# 创建随机森林分类器
model = se.RandomForestClassifier(max_depth=8,  # 最大树高
                                  n_estimators=150,  # 评估系数
                                  random_state=7)  # 随机种子
# 训练
model.fit(train_x, train_y)
print("accuracy:", model.score(train_x, train_y))  # 打印平均精度

# 平均精度
print('accuracy:',model.score(train_x,train_y))

# 特征重要性
print('Feature_importtance:',model.feature_importances_)

# 预测
## 待预测数据
data = [['high', 'med', '5more', '4', 'big', 'low'],
        ['high', 'high', '4', '4', 'med', 'med'],
        ['low', 'low', '2', '2', 'small', 'high'],
        ['low', 'med', '3', '4', 'med', 'high']]
data = np.array(data).T
test_x = []
for row in range(len(data)):
    encoder = encoders[row]  # 取得每列对应的标签编码器
    test_x.append(encoder.transform(data[row]))  # 待预测数据编码

test_x = np.array(test_x).T  # 转置回来

pred_test_y = model.predict(test_x)  # 执行预测

pred_test_y = encoders[-1].inverse_transform(pred_test_y)  # 预测结果反向编码
print("pred_test_y:", pred_test_y)  # 预测结果
