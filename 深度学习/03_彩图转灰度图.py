import cv2

# 读取原始图像
im = cv2.imread('./data/Linus.png',1)
cv2.imshow('im',im)

# 转成灰度图
im_gray = cv2.cvtColor(im,
                       cv2.COLOR_BGR2GRAY) # 转灰度图像
cv2.imshow('im_gray',im_gray) # 显示灰度图像

cv2.waitKey()
cv2.destroyAllWindows()