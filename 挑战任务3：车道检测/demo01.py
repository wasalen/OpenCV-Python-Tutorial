import cv2
import numpy as np

# 高斯滤波核大小
blur_ksize = 5
# Canny边缘检测高低阈值
canny_lth = 50
canny_hth = 150


def process_an_image(img):
    # 1. 灰度化、滤波和Canny
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
    edges = cv2.Canny(blur_gray, canny_lth, canny_hth)
    return edges


if __name__ == "__main__":
    img = cv2.imread('./test_pictures/lane.jpg')
    result = process_an_image(img)
    print(len(img), len(result))

    cv2.imshow("lane", result)
    cv2.waitKey(0)
