import cv2

a = cv2.imread('./data/lena.jpg', 0)
b = cv2.imread('./data/lily_square.png', 0)

c = cv2.imread('./data/3.png', 0)
d = cv2.imread('./data/4.png', 0)

dst1 = cv2.add(a, b)  # 图像直接相加，会导致图像过亮，过白

# 加权求和
dst2 = cv2.addWeighted(a, 0.6, b, 0.4, 0)  # 最后一个参数是亮度调节量
# 图像相减
dst3 = cv2.subtract(c,d)  # 图像相减，求出图像的差异

cv2.imshow("a", a)
cv2.imshow("b", b)
cv2.imshow("dst1", dst1)
cv2.imshow("dst2", dst2)

cv2.imshow('c', c)
cv2.imshow('d', d)
cv2.imshow('dst3',dst3)

cv2.waitKey()
cv2.destroyAllWindows()
