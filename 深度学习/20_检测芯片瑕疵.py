import cv2
import numpy as np
import math

# 第一步：图像预处理
## 1. 转换成灰度图像，进行二值化处理
im_cpu = cv2.imread("./data/CPU3.png")
im_gray = cv2.cvtColor(im_cpu, cv2.COLOR_BGR2GRAY)  # 转换成灰度图像

# 提取出度盘轮廓
ret, im_bin = cv2.threshold(im_gray, 162, 255, cv2.THRESH_BINARY)  # 图像二值化
cv2.imshow("im_cpu", im_cpu)
cv2.imshow("im_gray", im_gray)
cv2.imshow("im_bin", im_bin)

# 提取轮廓、绘制边沿
contours, hierarchy = cv2.findContours(im_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 绘制前景对象轮廓
mask = np.zeros(im_bin.shape, np.uint8)
mask = cv2.drawContours(mask, contours, -1, (255, 0, 0), -1)  # 绘制实心轮廓
cv2.imshow("mask", mask)

# 前景实心轮廓图和二值化图相减
im_sub = cv2.subtract(mask, im_bin)
cv2.imshow("im_sub", im_sub)

# 图像闭运算，先膨胀后腐蚀，去除内部毛刺
k = np.ones((10, 10), np.uint8)
im_close = cv2.morphologyEx(im_sub, cv2.MORPH_CLOSE, k, iterations=3)
cv2.imshow("im_close", im_close)

# 提取、绘制轮廓、计算面积
contours, hierarchy = cv2.findContours(im_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

(x, y), radius = cv2.minEnclosingCircle(contours[1])
center = (int(x), int(y))
radius = int(radius)
print("center:", center, " radius:", radius)
cv2.circle(im_close, center, radius, (255, 0, 0), 2)  # 绘制圆
cv2.imshow("im_gaussian_blur2", im_close)

# 在原始图片上绘制瑕疵
cv2.circle(im_cpu, center, radius, (0, 0, 255), 2)  # 绘制圆
cv2.imshow("im_cpu2", im_cpu)

# 计算面积
area = math.pi * radius * radius
print("area:", area)
if area > 12:
    print("度盘表面有缺陷")

cv2.waitKey()
cv2.destroyAllWindows()
