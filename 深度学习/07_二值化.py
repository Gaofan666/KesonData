import cv2

im = cv2.imread('./data/lena.jpg', 0)
cv2.imshow('im', im)

# 二值化  返回阈值和二值化后的图像
t, im_bin = cv2.threshold(im,
                          127, 255,  # 阈值和最大值  阈值越高白色的部分越少
                          cv2.THRESH_BINARY)  # 二值化选项
cv2.imshow('im_bin', im_bin)

# 反二值化  小于阈值部分设置为255
t, im_bin_inv = cv2.threshold(im,
                              127, 255,  # 阈值和最大值  阈值越高白色的部分越少
                              cv2.THRESH_BINARY_INV)  # 反二值化选项
cv2.imshow('im_bin_inv', im_bin_inv)

cv2.waitKey()
cv2.destroyAllWindows()
