from scipy import signal
from scipy import misc
import matplotlib.pyplot as mp
import numpy as np
import scipy.ndimage as sn
import imageio

# 读取图像
im = misc.imread('./data/zebra.png', flatten=True)
# 如果读不出就用下面这句
# im = sn.imread('./data/zebra.png',flatten=True)

# 定义卷积核
flt = np.array([[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]])
# 对图像进行卷积
conv_img1 = signal.convolve2d(im,  # 原图
                              flt,  # 卷积核
                              boundary='symm',  # 边沿处理方式
                              mode='same').astype('int32')  # same表示输入输出的矩阵一样大
# 显示原图和经过卷积的图像
mp.figure('Conv2d')
mp.subplot(1, 3, 1)  # 1行3列第1个子图
mp.imshow(im, cmap='gray')  # 原图，灰度
mp.xticks([])
mp.yticks([])

mp.subplot(1, 3, 2)  # 卷积后的图
mp.imshow(conv_img1, cmap='gray')  # 原图，灰度
mp.xticks([])
mp.yticks([])

mp.show()
