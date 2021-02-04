# 噪声密度聚类示例
import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp
import sklearn.metrics as sm

# 读取样本
x = []
with open("./data/perf.txt", "r") as f:
    for line in f.readlines():
        line = line.replace("\n", "")
        data = [float(substr) for substr in line.split(",")]
        x.append(data)

x = np.array(x)

epsilon = 0.8  # 邻域半径
min_samples = 5  # 最小样本数

# 创建噪声密度聚类器
model = sc.DBSCAN(eps=epsilon,  # 半径
                  min_samples=min_samples)  # 最小样本数
model.fit(x)
score = sm.silhouette_score(x,
                            model.labels_,
                            sample_size=len(x),
                            metric='euclidean')  # 计算轮廓系数
pred_y = model.labels_
print(pred_y) # 打印所有样本类别
# print(model.core_sample_indices_) # 打印所有核心样本索引

# 区分样本
core_mask = np.zeros(len(x), dtype=bool)
core_mask[model.core_sample_indices_] = True  # 核心样本下标

offset_mask = (pred_y == -1)  # 孤立样本
periphery_mask = ~(core_mask | offset_mask)  # 核心样本、孤立样本之外的样本

# 可视化
mp.figure('DBSCAN Cluster', facecolor='lightgray')
mp.title('DBSCAN Cluster', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=14)
mp.grid(linestyle=':')
labels = set(pred_y)
print(labels)
cs = mp.get_cmap('brg', len(labels))(range(len(labels)))
print("cs:", cs)

# 核心点
mp.scatter(x[core_mask][:, 0],  # x坐标值数组
           x[core_mask][:, 1],  # y坐标值数组
           c=cs[pred_y[core_mask]],
           s=80, label='Core')
# 边界点
mp.scatter(x[periphery_mask][:, 0],
           x[periphery_mask][:, 1],
           edgecolor=cs[pred_y[periphery_mask]],
           facecolor='none', s=80, label='Periphery')
# 噪声点
mp.scatter(x[offset_mask][:, 0],
           x[offset_mask][:, 1],
           marker='D', c=cs[pred_y[offset_mask]],
           s=80, label='Offset')
mp.legend()
mp.show()