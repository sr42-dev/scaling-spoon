# Warp perspective - bird's eye perspective of images

import cv2
import numpy as np

img = cv2.imread('/home/sr42/Projects/scaling-spoon/day2/cards.jpg')

width, height = 250, 350
pts1 = np.float32([[123, 55],[190, 60],[117, 158],[194, 156]]) # to get an adobe scan like image. Points indicate corner points of original document (random here)
pts2 = np.float32([[0, 0],[width, 0],[0, height],[width, height]]) # points indicate output image coordinates

matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgOutput = cv2.warpPerspective(img, matrix, (width,height))


cv2.imshow('Image', img)
cv2.imshow('Output', imgOutput)

cv2.waitKey(0)