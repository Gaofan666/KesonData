import cv2
import numpy as np

im = cv2.imread('./data/paper.jpg')
gray = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)
cv2.imshow('im', im)

# 模糊
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# 膨胀
dilate = cv2.dilate(blurred,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # 根据函数返回kenerl

# 检测边沿
edged = cv2.Canny(dilate,  # 原始图像
                  30, 120,   # 滞后阈值，模糊度
                  3) # 边缘厚度

# 轮廓检测
cnts, hie = cv2.findContours(edged.copy(),
                             cv2.RETR_EXTERNAL,  # 只检测外轮廓
                             cv2.CHAIN_APPROX_SIMPLE)  # 只保留该方向的终点坐标

docCnt = None

# 绘制轮廓
im_cnt = cv2.drawContours(im,
                          cnts,  # 轮廓点列表
                          -1,  # 绘制全部轮廓
                          (0, 0, 255),  # 轮廓颜色 红色
                          2)  # 轮廓粗细
cv2.imshow('im_cnt', im_cnt)

# 计算轮廓面积，并排序
if len(cnts) > 0:
    cnts = sorted(cnts,
                  key=cv2.contourArea,  # 排序依据，根据contourArea函数结果排序
                  reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c, True)  # 计算罗阔周长
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # 轮廓多边形拟合
        # 轮廓为4个点表示找到智障
        if len(approx) == 4:
            docCnt = approx
            break
print(docCnt)

# 用圆圈标记处角点
points = []
for peak in docCnt:
    peak = peak[0]
    # 画圆
    cv2.circle(im,
               tuple(peak), 10,  # 圆心  半径
               (0, 0, 255), 2)  # 颜色 粗细
    points.append(peak)
print(points)
cv2.imshow('im_point', im)

# 校正
src = np.float32([points[0], points[1], points[2], points[3]])  # 原来逆时针方向四个点
dst = np.float32([[0, 0], [0, 488], [337, 488], [337, 0]])  # 对应变换后逆时针方向四个点
m = cv2.getPerspectiveTransform(src, dst)  # 生成透视变换矩阵
result = cv2.warpPerspective(gray.copy(), m, (337, 488))
cv2.imshow('result', result)  # 显示结果

cv2.waitKey()
cv2.destroyAllWindows()
