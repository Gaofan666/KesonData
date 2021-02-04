import numpy as np
import sklearn.model_selection as ms
import sklearn.naive_bayes as nb
import matplotlib.pyplot as mp

x, y = [], []  # 输入，输出

# 读取数据文件
with open("./data/multiple1.txt", "r") as f:
    for line in f.readlines():
        data = [float(substr) for substr in line.split(",")]
        x.append(data[:-1])  # 输入样本：取从第一列到导数第二列
        y.append(data[-1])  # 输出样本：取最后一列

x = np.array(x)
y = np.array(y, dtype=int)

# 定义模型
model = nb.GaussianNB()
# 交叉验证
pw = ms.cross_val_score(model,
                        x, y,
                        cv=5,  # 折叠数量
                        scoring='precision_weighted'
                        )
print(pw)

f1 = ms.cross_val_score(model,
                        x, y,
                        cv=5,  # 折叠数量
                        scoring='f1_weighted'
                        )
print(f1.mean())

acc = ms.cross_val_score(model,
                         x, y,
                         cv=5,  # 折叠数量
                         scoring='accuracy'  # 准确率
                         )
print(acc.mean())
