# ex2tron's blog:
# http://ex2tron.wang

import cv2
import numpy as np
from matplotlib import pyplot as plt


# 1.模板匹配
img = cv2.imread('lena.jpg', 0)
template = cv2.imread('face.jpg', 0)
h, w = template.shape[:2]  # rows->h, cols->w

# 6种匹配方法
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

"""
平方差匹配CV_TM_SQDIFF：用两者的平方差来匹配，最好的匹配值为0
归一化平方差匹配CV_TM_SQDIFF_NORMED
相关匹配CV_TM_CCORR：用两者的乘积匹配，数值越大表明匹配程度越好
归一化相关匹配CV_TM_CCORR_NORMED
相关系数匹配CV_TM_CCOEFF：用两者的相关系数匹配，1表示完美的匹配，-1表示最差的匹配
归一化相关系数匹配CV_TM_CCOEFF_NORMED
"""

for meth in methods:
    img2 = img.copy()

    # 匹配方法的真值
    method = eval(meth)
    res = cv2.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # 如果是平方差匹配TM_SQDIFF或归一化平方差匹配TM_SQDIFF_NORMED，取最小值
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # 画矩形
    cv2.rectangle(img2, top_left, bottom_right, 255, 2)

    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
    plt.subplot(122), plt.imshow(img2, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()


# 2.匹配多个物体
img_rgb = cv2.imread('mario.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('mario_coin.jpg', 0)
h, w = template.shape[:2]

res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
# 取匹配程度大于%80的坐标
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):  # *号表示可选参数
    bottom_right = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img_rgb, pt, bottom_right, (0, 0, 255), 2)

cv2.imshow('img_rgb', img_rgb)
cv2.waitKey(0)


# 3.有关几个函数的说明：
x = np.arange(9.).reshape(3, 3)
print(np.where(x > 5))
# 结果：(array([2, 2, 2]), array([0, 1, 2]))

x = [1, 2, 3]
y = [4, 5, 6]

print(list(zip(x, y, x)))  # [(1, 4), (2, 5), (3, 6)]
