import numpy as np
import cv2

im = cv2.imread('./data/Linus.png')
cv2.imshow('im', im)

# 缩小
h, w = im.shape[:2]
dst_size = (int(w / 2), int(h / 2))  # 取出高度宽度
im_resized1 = cv2.resize(im, dst_size)  # k宽度高度各一半
cv2.imshow('reduce', im_resized1)

# 放大
dst_size = (200, 300)  # 放大后的图像大小
im_resized2 = cv2.resize(im,
                         dst_size,
                         interpolation=cv2.INTER_NEAREST)  # 最邻近插值
cv2.imshow('im_resized2', im_resized2)

im_resized3 = cv2.resize(im,dst_size,
                         interpolation=cv2.INTER_LINEAR) # 双线性插值
cv2.imshow('im_resized3', im_resized3)

cv2.waitKey()
cv2.destroyAllWindows()
