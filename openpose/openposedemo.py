# -*- coding:utf-8 -*- by jiaming
import cv2

protoFile = "pose_deploy_linevec.prototxt"
weightsFile = "pose_iter_440000.caffemodel"

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# 读取图像
frame = cv2.imread("2.jpg")
frameW = frame.shape[1]
frameH = frame.shape[0]

# 指定输入图像的尺寸
inWidth = 368
inHeight = 368

# 将图片输入网络
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

# 设置网络
net.setInput(inpBlob)
output = net.forward()

H = output.shape[2]
W = output.shape[3]
# 空列表来存储检测到的关键点
points = []
popo = []
for i in range(14):  # in range(output.shape[1]):#in range(14):#这边需要修改
    # 对应主体部分的置信图。
    probMap = output[0, i, :, :]

    # 找到分布图的全局最大值
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    # 按比例缩放以适应原始图像。
    x = (frameW * point[0]) / W
    y = (frameH * point[1]) / H

    if prob > 0.1:
        cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3,
                    lineType=cv2.LINE_AA)
        # 如果概率大于阈值，则将点添加到列表中d
        points.append((int(x), int(y)))
print(points)
# for i in range()://测试后需要改进
cv2.line(frame, points[0], points[1], (0, 255, 0), 3)
cv2.line(frame, points[1], points[2], (0, 255, 0), 3)
cv2.line(frame, points[2], points[3], (0, 255, 0), 3)
cv2.line(frame, points[3], points[4], (0, 255, 0), 3)

cv2.line(frame, points[1], points[5], (0, 255, 0), 3)
cv2.line(frame, points[5], points[6], (0, 255, 0), 3)
cv2.line(frame, points[6], points[7], (0, 255, 0), 3)
cv2.line(frame, points[1], points[8], (0, 255, 0), 3)
cv2.line(frame, points[8], points[9], (0, 255, 0), 3)
cv2.line(frame, points[9], points[10], (0, 255, 0), 3)
cv2.line(frame, points[1], points[11], (0, 255, 0), 3)
cv2.line(frame, points[11], points[12], (0, 255, 0), 3)
cv2.line(frame, points[12], points[13], (0, 255, 0), 3)

cv2.namedWindow("Output-Keypoints", 2)
cv2.imshow("Output-Keypoints", frame)
cv2.waitKey()
cv2.destroyAllWindows()
