# ex2tron's blog:
# http://ex2tron.wang

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1.Canny边缘检测
img = cv2.imread('666.jpg', 0)
edges = cv2.Canny(img, 30, 70)

# cv2.imshow('canny', np.hstack((img, edges)))
ret, th1 = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow('canny',  th1)
# cv2.waitKey(0)
# cv2.imwrite("a.jpg", edges)


# 2.先阈值，后边缘检测
# 阈值分割（使用到了番外篇讲到的Otsu自动阈值）
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
edges = cv2.Canny(thresh, 30, 70)
# cv2.imshow('canny', thresh)
cv2.imshow('canny', np.hstack((img, thresh, edges)))
cv2.waitKey(0)
