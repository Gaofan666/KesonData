import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = tf.linspace(-8., 8, 100)  # 设置x的坐标间隔
y = tf.linspace(-8., 8, 100)  # 设置x的坐标间隔
x, y = tf.meshgrid(x, y)  # 生成网格点，并拆分后返回
z = tf.sqrt(x*2-x**2)

fig = plt.figure()
ax = Axes3D(fig)
# 根据网格点绘制sinc函数3D曲线
ax.contour3D(x.numpy(), y.numpy(), z.numpy(), 50)
plt.show()