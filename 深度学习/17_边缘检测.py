# 边沿检测示例
import cv2 as cv
import numpy as np

im = cv.imread('./data/lily.png', 0)
cv.imshow('Original', im)

# # 水平方向滤波
# hsobel = cv.Sobel(im, cv.CV_64F, 1, 0, ksize=5)
# cv.imshow('H-Sobel', hsobel)
# # 垂直方向滤波
# vsobel = cv.Sobel(im, cv.CV_64F, 0, 1, ksize=5)
# cv.imshow('V-Sobel', vsobel)
# 两个方向滤波
"""
cv.CV_64F: 深度，应该设置为-1，可能发生计算错误，所以设置为精度更高的CV-64F
"""
# sobel = cv.Sobel(im, cv.CV_64F, 1, 1, ksize=5)
# print(type(sobel))
# print(sobel)
# cv.imshow('Sobel', sobel)
#
# # Laplacian滤波：对细节反映更明显
# laplacian = cv.Laplacian(im, cv.CV_64F)
# cv.imshow('Laplacian', laplacian)
#
# # Canny边沿提取
# canny = cv.Canny(im,
#                  50, # 滞后阈值
#                  240) # 模糊度
# cv.imshow('Canny', canny)

# # 中值滤波
# median = cv.medianBlur(im,
#                        5  # 滤波器核大小(区域)
#                        )
# cv.imshow('Median', median)
#
# # 均值滤波
# mean = cv.blur(im,
#                (3, 3)  # 卷积核大小
#                )
# cv.imshow("Mean", mean)

# 高斯滤波
gaussian = cv.GaussianBlur(im, (5, 5),  # 卷积核大小
                           3,  # 标准差
                           )
cv.imshow("Gaussian", gaussian)

# 自己定义一个高斯模糊滤波器
gau_flt = np.array([
    [1, 4, 7, 4, 1],
    [4, 16, 26, 16, 4],
    [7, 26, 41, 26, 7],
    [4, 16, 26, 16, 4],
    [1, 4, 7, 4, 1]
], np.float32)/273 # 除以一个273是为了防止卷积核太大，图片变成白色

gau_conv2d = cv.filter2D(im,
                         -1,  # 图像的深度（通道数） -1代表与原始图像一致
                         gau_flt  # 卷积核
                         )
cv.imshow("gau_conv2d",gau_conv2d)

cv.waitKey()
cv.destroyAllWindows()
