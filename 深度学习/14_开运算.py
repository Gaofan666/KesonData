# 先腐蚀，后膨胀
import cv2
import numpy as np

# 读取原图
im1 = cv2.imread('./data/7.png')
im2 = cv2.imread('./data/8.png')

# 开运算
k = np.ones((10, 10), np.uint8)  # 计算核的类型uint8
r1 = cv2.morphologyEx(im1, cv2.MORPH_OPEN, k)

r2 = cv2.morphologyEx(im2, cv2.MORPH_OPEN, k)

cv2.imshow('im1', im1)
cv2.imshow('r1', r1)
cv2.imshow('im2', im2)
cv2.imshow('r2', r2)

cv2.waitKey()
cv2.destroyAllWindows()
