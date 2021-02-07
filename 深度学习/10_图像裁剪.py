import numpy as np
import cv2


# 图像随机裁剪
def random_crop(im, w, h):
    start_x = np.random.randint(0, im.shape[1])
    start_y = np.random.randint(0, im.shape[0])
    # 通过切片实现图像裁剪
    new_img = im[start_y:start_y + h, start_x:start_x + w]
    return new_img


# 中心裁剪
def center_crop(im, w, h):
    start_x = int(im.shape[1] / 2) - int(w / 2)  # 中心的图像坐标剪掉半个宽度
    start_y = int(im.shape[0] / 2) - int(h / 2)  # 中心的图像坐标剪掉半个宽度
    # 通过切片实现图像裁剪
    new_img = im[start_y:start_y + h, start_x:start_x + w]
    return new_img


if __name__ == '__main__':
    im = cv2.imread('./data/banana_1.png')
    # 随机裁剪
    new_img = random_crop(im, 200, 200)
    cv2.imshow('random_cop', new_img)

    # 中心裁剪
    new_img2 = center_crop(im, 200, 200)
    cv2.imshow('center_crop', new_img2)

    cv2.waitKey()
    cv2.destroyAllWindows()
