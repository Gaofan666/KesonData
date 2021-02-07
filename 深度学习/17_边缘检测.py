# 边沿检测示例
import cv2 as cv

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
sobel = cv.Sobel(im, cv.CV_64F, 1, 1, ksize=5)
cv.imshow('Sobel', sobel)

# Laplacian滤波：对细节反映更明显
laplacian = cv.Laplacian(im, cv.CV_64F)
cv.imshow('Laplacian', laplacian)

# Canny边沿提取
canny = cv.Canny(im,
                 50, # 滞后阈值
                 240) # 模糊度
cv.imshow('Canny', canny)

cv.waitKey()
cv.destroyAllWindows()