import cv2
import numpy as np

# 读取原始图像
im = cv2.imread('./data/lena.jpg', 0)
cv2.imshow('im', im)

# 中值滤波
im_median = cv2.medianBlur(im, 5)  # 中值滤波大小
cv2.imshow('im_md', im_median)

# 均值滤波
im_mean = cv2.blur(im,(3,3)) # （3，3）为kernel大小
cv2.imshow('im_me',im_mean)

# 高斯滤波
im_guassian = cv2.GaussianBlur(im,
                               (5,5), # kernel大小
                               3) # 标准差
cv2.imshow('im_gua',im_guassian)

cv2.waitKey()
cv2.destroyAllWindows()
