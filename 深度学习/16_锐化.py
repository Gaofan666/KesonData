# 瑞华：增大像素之间的差异

import cv2
import numpy as np

im = cv2.imread('./data/lena.jpg', 0)
cv2.imshow('im', im)

# 锐化算子1
sharpen_1 = np.array([[-1, -1, -1],
                      [-1, 9, -1],
                      [-1, -1, -1]])

# 执行滤波计算
im_sharpen1 = cv2.filter2D(im,
                           -1,  # 深度（所有通道）
                           sharpen_1)
cv2.imshow('im_s1', im_sharpen1)

# 瑞华算子2
sharpen_2 = np.array([[0, -1, 0],
                      [-1, 8, -1],
                      [0, 1, 0]]) / 4.0
im_sharpen2 = cv2.filter2D(im, -1, sharpen_2)
cv2.imshow('im_s2', im_sharpen2)

cv2.waitKey()
cv2.destroyAllWindows()
