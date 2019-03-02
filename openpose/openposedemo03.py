# -*- coding:utf-8 -*- by jiaming
import cv2
import numpy as np
import matplotlib.pyplot as plt

protoFile = "pose_deploy_linevec.prototxt"
weightsFile = "pose_iter_440000.caffemodel"

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

image1 = cv2.imread("group.jpg")
# Fix the input Height and get the width according to the Aspect Ratio
inHeight = 368
inWidth = int((inHeight / frameHeight) * frameWidth)

inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                                (0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)
output = net.forward()

i = 0
probMap = output[0, i, :, :]
probMap = cv2.resize(probMap, (frameWidth, frameHeight))

plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
plt.imshow(probMap, alpha=0.6)

mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)
mapMask = np.uint8(mapSmooth > threshold)

# find the blobs
_, contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# for each blob find the maxima
for cnt in contours:
    blobMask = np.zeros(mapMask.shape)
    blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
    maskedProbMap = mapSmooth * blobMask
    _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
    keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

pafA = output[0, mapIdx[k][0], :, :]
pafB = output[0, mapIdx[k][1], :, :]
pafA = cv2.resize(pafA, (frameWidth, frameHeight))
pafB = cv2.resize(pafB, (frameWidth, frameHeight))

# Find the keypoints for the first and second limb
candA = detected_keypoints[POSE_PAIRS[k][0]]
candB = detected_keypoints[POSE_PAIRS[k][1]]

d_ij = np.subtract(candB[j][:2], candA[i][:2])
norm = np.linalg.norm(d_ij)
if norm:
    d_ij = d_ij / norm
# Find p(u)
interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
                        np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
# Find L(p(u))
paf_interp = []
for k in range(len(interp_coord)):
    paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                       pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
# Find E
paf_scores = np.dot(paf_interp, d_ij)
avg_paf_score = sum(paf_scores)/len(paf_scores)

# Check if the connection is valid
# If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
    if avg_paf_score > maxScore:
        max_j = j
        maxScore = avg_paf_score
for j in range(len(personwiseKeypoints)):
    if personwiseKeypoints[j][indexA] == partAs[i]:
        person_idx = j
        found = 1
        break

if found:
    personwiseKeypoints[person_idx][indexB] = partBs[i]
# if find no partA in the subset, create a new subset
elif not found and k < 17:
    row = -1 * np.ones(19)
    row[indexA] = partAs[i]
    row[indexB] = partBs[i]

for i in range(17):
    for n in range(len(personwiseKeypoints)):
        index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
        if -1 in index:
            continue
        B = np.int32(keypoints_list[index.astype(int), 0])
        A = np.int32(keypoints_list[index.astype(int), 1])
        cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), colors[i], 2, cv2.LINE_AA)

cv2.imshow("Detected Pose", frameClone)
cv2.waitKey(0)
