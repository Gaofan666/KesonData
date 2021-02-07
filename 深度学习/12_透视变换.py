import numpy as np
import cv2

im = cv2.imread('./data/pers.png')
h, w = im.shape[:2]  # 取出高度和宽度

# 设置透视变换的原点和目标点
pts1 = np.float32([[58, 2], [167, 9], [8, 196], [126, 196]])  # 输入图像四个顶点坐标
pts2 = np.float32([[16, 2], [167, 8], [8, 196], [169, 196]])  # 输出图像四个顶点坐标

# 生成透视变换矩阵
M = cv2.getPerspectiveTransform(pts1,  # 输入图像四个顶点坐标
                                pts2)  # 输出图像四个顶点坐标

# 执行透视变换
dst = cv2.warpPerspective(im, M, (w, h))

M = cv2.getPerspectiveTransform(pts2,  # 输chu图像四个顶点坐标
                                pts1)  # 输ru图像四个顶点坐标

dst2 = cv2.warpPerspective(dst,  # 矫正过的图像
                           M,
                           (w, h))

cv2.imshow('im', im)
cv2.imshow('dst', dst)
cv2.imshow('dst2', dst2)

cv2.waitKey()
cv2.destroyAllWindows()
