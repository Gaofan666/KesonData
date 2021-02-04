import cv2
import matplotlib.pyplot as plt

# 读取原图
im = cv2.imread('./data/sunrise.jpg', 0)
cv2.imshow('im', im)

# 直方图均衡化
im_equ = cv2.equalizeHist(im)
cv2.imshow('im_equ', im_equ)

# 绘制直方图
plt.subplot(2, 1, 1)  # 2行1列第一个子图
arr = im.ravel()  # 扁平化，根据灰度统计像素出现次数
plt.hist(arr,  # 统计数据
         256,  # 柱体的数量
         [0, 255],  # 范围
         label='orig')  # 显示文字，原图
# 均衡化后的统计直方图
plt.subplot(2, 1, 2)
plt.hist(im_equ.ravel(), 255, [0, 255], label='equ')

plt.show()

cv2.waitKey()
cv2.destroyAllWindows()
