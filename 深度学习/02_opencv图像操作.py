# 图像读取，显示，保存

import cv2

# 读取
im = cv2.imread('./data/Linus.png',
                1)  # 读取方式 0-单通道图像 1-彩色图像

print(type(im))  # ndarray
print(im.shape)  # 打印形状

# 显示图像
cv2.imshow('im',  # 显示图像的窗体名称，不能重复
           im)  # 图像数据

# 保存图像
cv2.imwrite('./Linus_new.png', im)

cv2.waitKey()  # 等待用户敲击按键（阻塞函数）
cv2.destroyAllWindows()  # 销毁所有创建的窗体
