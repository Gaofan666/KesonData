# 查找图像轮廓
import cv2
import numpy as np

im = cv2.imread("./data/3.png")
cv2.imshow("orig", im)

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# 图像二值化处理，将大于阈值的设置为最大值，其它设置为0
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 查找图像边沿：cv2.findContours
contours, hierarchy = cv2.findContours(binary,  # 二值化处理后的图像
                                            cv2.RETR_EXTERNAL,  # 只检测外轮廓
                                            cv2.CHAIN_APPROX_NONE)  # 存储所有的轮廓点
# 打印所有轮廓值
for cnt in contours:
    print(cnt.shape)

# 绘制边沿
im_cnt = cv2.drawContours(im,  # 绘制图像
                          contours,  # 轮廓点列表
                          -1,  # 绘制全部轮廓
                          (0, 0, 255),  # 轮廓颜色：红色
                          2)  # 轮廓粗细
cv2.imshow("im_cnt", im_cnt)

cv2.waitKey()
cv2.destroyAllWindows()