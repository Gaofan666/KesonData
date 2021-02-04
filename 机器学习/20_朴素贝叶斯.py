import numpy as np
import sklearn.naive_bayes as nb
import matplotlib.pyplot as mp

# 读取数据
x, y = [], []
with open('./data/multiple1.txt', 'r') as f:
    for line in f.readlines():
        data = [float(substr) for substr in line.split(',')]
        x.append(data[:-1])  # 输入部分  切出1，2列
        y.append(data[-1])  # 输出部分

x = np.array(x)
y = np.array(y, dtype=int) # 因为输出是类别，所以是整型

# 定义模型
model = nb.GaussianNB()
model.fit(x,y)

# 计算显示范围
left = x[:, 0].min() - 1
right = x[:, 0].max() + 1

buttom = x[:, 1].min() - 1
top = x[:, 1].max() + 1

grid_x, grid_y = np.meshgrid(np.arange(left, right, 0.01),
                             np.arange(buttom, top, 0.01))

mesh_x = np.column_stack((grid_x.ravel(), grid_y.ravel()))
mesh_z = model.predict(mesh_x)
mesh_z = mesh_z.reshape(grid_x.shape)

mp.figure('Naive Bayes Classification', facecolor='lightgray')
mp.title('Naive Bayes Classification', fontsize=20)
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=10)
mp.pcolormesh(grid_x, grid_y, mesh_z, cmap='gray')
mp.scatter(x[:, 0], x[:, 1], c=y, cmap='brg', s=80)
mp.show()
