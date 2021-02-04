import cv2
import matplotlib.pyplot as plt

im = cv2.imread('./data/sunrise.jpg')  # 默认读取彩色图像
cv2.imshow('im', im)

# BGR--YUV亮度空间
yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
# 取出亮度通道。均衡化处理后覆盖原亮度通道
# ...表示所有行方向列方向
yuv[..., 0] = cv2.equalizeHist(yuv[..., 0])
# cv2.imshow('yuv', yuv)
im_equ = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)  # 转回来
cv2.imshow('im_equ', im_equ)

cv2.waitKey()
cv2.destroyAllWindows()
