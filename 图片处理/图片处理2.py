import cv2
import numpy as np

num_down = 2  # 减少像素的数目
num_bilateral = 7  # 定义双边波的数目

img_rgb = cv2.imread("./666.jpg")

img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(img_hsv)
# 直方图均衡化
v = cv2.equalizeHist(v)
img_hsv = cv2.merge((h, v, s))
# 中值滤波
img_hsv = cv2.medianBlur(img_hsv, 7)
# 形态学变换-开运算
kernel = np.ones((5, 5), np.uint8)
img_hsv = cv2.morphologyEx(img_hsv, cv2.MORPH_OPEN, kernel, iterations=3)
# 中值滤波
img_hsv = cv2.medianBlur(img_hsv, 7)
# print(img_hsv)
# cv2.imshow(img_hsv)
# cv2.waitKey(0)
img_mask = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
img_mask = cv2.cvtColor(img_mask, cv2.COLOR_RGB2GRAY)
img_mask = cv2.medianBlur(img_mask, 7)

#

# 创建轮廓
# 检测到边缘并增强其效果
img_edge = cv2.adaptiveThreshold(img_mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)
# cv2.imshow('cattoon', img_edge)
# cv2.waitKey(0)
#  合并轮廓与彩色图片

# 转换彩色图像
img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)
# print(len(img_color))
img_cartoon = cv2.bitwise_and(img_rgb, img_edge)

# 显示图片

cv2.imshow('cattoon', img_edge)
cv2.waitKey(0)
