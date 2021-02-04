import cv2

im = cv2.imread('./data/opencv2.png', 1)
cv2.imshow('im', im)

# 利用切片操作，取出蓝色通道  BGR-012
b = im[:, :, 0]  # 所有行，所有列  0是第一个通道为蓝色通道
cv2.imshow('blue', b)

# 抹掉蓝色通道
im[:, :, 0] = 0  # 将蓝色通道的值全置为0
cv2.imshow('im_b0', im)

# 抹掉绿色通道
im[:, :, 1] = 0  # 将绿色通道的值全置为0
cv2.imshow('im_b1', im)

cv2.waitKey()
cv2.destroyAllWindows()
