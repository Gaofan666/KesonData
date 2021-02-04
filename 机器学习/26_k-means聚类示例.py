import numpy as np
import sklearn.cluster as sc
import matplotlib.pyplot as mp
import sklearn.metrics as sm

x = []
with open('./data/multiple3.txt','r') as f:
    for line in f.readlines():
        line = line.replace('\n','')
        data = [float(substr) for substr in line.split(',')]
        x.append(data)
x = np.array(x) # 列表转数组

# 定义模型、
model = sc.KMeans(n_clusters=4) # n_clusters为聚类数量
model.fit(x) # 训练（执行聚类计算# ）
pred_y = model.labels_  # 聚类结果
centers = model.cluster_centers_ # 聚类中心

print('聚类结果：\n',pred_y,'\n')
print('聚类中心：',centers)

# 可视化
mp.figure('k-means')
mp.title('k-means')
mp.xlabel('x',fontsize=14)
mp.ylabel('y',fontsize=14)
mp.tick_params(labelsize=10)
mp.scatter(x[:,0],x[:,1],s=80,c=pred_y,cmap='brg')
# 绘制中心点
mp.scatter(centers[:,0],centers[:,1],marker='+',c='black',s=200,linewidths=1)
mp.show()