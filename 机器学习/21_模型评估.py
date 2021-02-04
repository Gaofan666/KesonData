# 混淆矩阵示例
import numpy as np
import sklearn.model_selection as ms
import sklearn.metrics as sm
import sklearn.naive_bayes as nb

# 输入，输出
x, y = [], []

# 读取数据文件
with open("./data/multiple1.txt", "r") as f:
    for line in f.readlines():
        data = [float(substr) for substr in line.split(",")]
        x.append(data[:-1])  # 输入样本：取从第一列到导数第二列
        y.append(data[-1])  # 输出样本：取最后一列

# 样本转数组
x = np.array(x)
y = np.array(y, dtype=int)

# 划分训练集和测试集
train_x, test_x, train_y, test_y = ms.train_test_split(
    x, y, test_size=0.25, random_state=7)

# 创建高斯朴素贝叶斯分类器对象
model = nb.GaussianNB()
model.fit(train_x, train_y)  # 使用划分的训练集来训练模型
pred_test_y = model.predict(test_x)  # 预测

print("recall:", sm.recall_score(test_y,  # 真实值
                                 pred_test_y,  # 预测值
                                 average="macro"))  # 计算平均值，不考虑样本权重
print("precision:", sm.precision_score(test_y,  # 真实值
                                       pred_test_y,  # 预测值
                                       average="macro"))  # 计算平均值，不考虑样本权重
print("F1:", sm.f1_score(test_y, pred_test_y,average="macro"))

# 计算并打印模型预测的混淆矩阵
print("\n Confusion Matrix:")
cm = sm.confusion_matrix(test_y, pred_test_y)
print(cm)