import cv2

num_down = 2  # 减少像素的数目
num_bilateral = 7  # 定义双边波的数目

img_rgb = cv2.imread("./1236.jpg")

# 用高斯金字塔降低取样
img_color = img_rgb
for _ in range(num_down):
    img_color = cv2.pyrDown(img_color)


# 重复使用小的双边滤波代替一个大的滤波
for _ in range(num_bilateral):
    img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=7, sigmaSpace=2)
# print(len(img_color))
# 升采样图片到原始大小

for _ in range(num_down):
    img_color = cv2.pyrUp(img_color)

# 转换灰度，并使用中值滤波器减少噪点
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
img_blur = cv2.medianBlur(img_gray, 7)

# 创建轮廓
# 检测到边缘并增强其效果
img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=5)
# cv2.imshow('cattoon', img_edge)
# cv2.waitKey(0)
#  合并轮廓与彩色图片

# 转换彩色图像
img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)
# print(len(img_color))
img_cartoon = cv2.bitwise_and(img_color, img_edge)

# 显示图片

cv2.imshow('cattoon', img_edge)
cv2.waitKey(0)
cv2.imwrite('meng.jpg', img_edge)