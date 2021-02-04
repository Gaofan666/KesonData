import numpy as np
import cv2


def translate(img, x, y):
    """
    平移变换
    :param img: 原图
    :param x: 水平方向移动的像素
    :param y: 垂直方向移动的像素
    :return: 平移后的图像
    """
    h, w = img.shape[:2]  # 取出图像高度和宽度
    # 定义平移矩阵
    M = np.float32([[1, 0, x],
                    [0, 1, y]])
    # 调用函数实现仿射变换
    shifted = cv2.warpAffine(img, M, (w, h))
    return shifted


def rotate(img, angle, center=None, scale=1.0):
    """
    旋转变换
    :param img: 原图
    :param angle: 旋转角度
    :param center: 中心
    :param scale: 缩放的比例
    :return: 旋转后的图像
    """
    h, w = img.shape[:2]
    # 如果中心未设置None，以原图的中心为中心
    if center is None:
        center = (w / 2, h / 2)
    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, scale)
    # 使用矩阵进行仿射变换
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated


if __name__ == '__main__':
    # 读取原始图像
    im = cv2.imread('./data/Linus.png')
    cv2.imshow('im', im)
    # 平移
    shifted = translate(im, -20, 50)
    cv2.imshow('shifted', shifted)

    # 旋转
    h, w = im.shape[:2]
    print(h, w)
    cen = (h / 3, w / 4)
    print(type(cen))
    rotated = rotate(im,-45,center=cen) # 指定中心旋转
    cv2.imshow('rotated', rotated)

    cv2.waitKey()
    cv2.destroyAllWindows()
