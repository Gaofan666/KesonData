import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp

x = []  #
with open('./data/multiple3.txt', 'r') as f:
    for line in f.readlines():
        line = line.replace('\n', '')
        data = [float(substr) for substr in line.split(',')]
        x.append(data)
x = np.array(x)  # 转数组

model = sc.AgglomerativeClustering(n_clusters=4)
model.fit(x)
pred_y = model.labels_ # 取出聚类结果

# 可视化
mp.figure("Agglomerative", facecolor="lightgray")
mp.title("Agglomerative")
mp.xlabel("x", fontsize=14)
mp.ylabel("y", fontsize=14)
mp.tick_params(labelsize=10)
mp.scatter(x[:, 0], x[:, 1], s=80, c=pred_y, cmap="brg")
mp.show()
