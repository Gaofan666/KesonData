import cv2
import numpy as np

im = cv2.imread('./data/5.png')
cv2.imshow('im', im)

# 腐蚀
kernel = np.ones((3, 3), np.uint8)  # 计算腐蚀核
im_erode = cv2.erode(im,
                     kernel,
                     iterations=2) # 迭代次数
cv2.imshow('im_erode', im_erode)

cv2.waitKey()
cv2.destroyAllWindows()
