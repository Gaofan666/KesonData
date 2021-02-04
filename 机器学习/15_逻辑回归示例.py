import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as mp

x = np.array([[3, 1], [2, 5], [1, 8], [6, 4],
              [5, 2], [3, 5], [4, 7], [4, -1]])  # 样本点
y = np.array([0, 1, 1, 0, 0, 1, 1, 0]) # 类别

# 创建逻辑回归器
model = lm.LogisticRegression(solver='liblinear')  # 消除警告，版本问题
model.fit(x, y)
# 预测
test_x = np.array([[3, 9], [6, 1]])
test_y = model.predict(test_x)
print(test_y)

# 计算绘图坐标边界
left = x[:, 0].min() - 1  # 输入数据的第一列
right = x[:, 0].max() + 1  # 右边界
buttom = x[:, 1].min() - 1  # 计算下边界
top = x[:, 1].max() + 1

# 产生网格化矩阵
grid_x, grid_y = np.meshgrid(
    np.arange(left, right, 0.01),  # 产生均匀的x坐标
    np.arange(buttom, top, 0.01)  # 产生均匀的y坐标
)
print(grid_x.shape)
print(grid_y.shape)

# 先把x,y拉成1维 再合并成两列
mesh_x = np.column_stack((grid_x.ravel(), grid_y.ravel()))
print(mesh_x.shape)

# 将产生的网格矩阵坐标点，送入模型进行预测
mesh_z = model.predict(mesh_x)
mesh_z = mesh_z.reshape(grid_x.shape)  # 还原成2维

mp.figure('Logistic Regression')
mp.title('Logistic Regression')
mp.xlabel('x', fontsize=14)
mp.ylabel('y', fontsize=14)
mp.tick_params(labelsize=14)
mp.pcolormesh(grid_x, grid_y, mesh_z, cmap='gray')  # 给网格涂颜色

mp.scatter(x[:, 0], x[:, 1], c=y, cmap='brg', s=80)
mp.scatter(test_x[:, 0], test_x[:, 1], c='red', marker='s', s=80)

mp.show()
