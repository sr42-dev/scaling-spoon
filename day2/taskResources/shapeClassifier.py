# Contour / shape detection - left off at 1:30:26

import cv2
import numpy as np

def getContours(img) :

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # image, retrieval method (here, for outermost contours), approximation

    for cnt in contours : # contours - array of contours detected in image

        area = cv2.contourArea(cnt) # finds area of selected contour
        # print(area)
        cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3) # image copy, selected contour, (-1 to draw all contours), color, thickness
        if area > 500 : # selects only contours without too much noise (contours with area > 500 units)
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0),3)  # image copy, selected contour, (-1 to draw all contours), color, thickness
            perimeter = cv2.arcLength(cnt, True) # contour, is closed(?)
            # print(perimeter)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True) # contour, resolution, is closed(?)
            # print(approx) # gives corner points for each contour (shape)
            print(len(approx)) # prints number of vertices
            objCor = len(approx) # number of corners (higher the number, more likely to be a circle)
            x, y, w, h = cv2.boundingRect(approx) # coordinates of each shape

            if objCor == 4 : # evaluating number of corners to determine shape
                aspRatio = w / float(h)
                if aspRatio > 0.95 and aspRatio < 1.05 : # if width = height within error margin, then square
                    objType = 'Square'
                else :
                    objType = 'Quadrilateral'
            elif objCor == 5 :
                objType = 'Pentagon'
            elif objCor > 5 : # circle if large number of corners are detected (large being greater than 5 here)
                objType = 'Conic'
            else :
                objType = 'None'

            cv2.rectangle(imgContour, (x,y), (x+w, y+h), (0, 255, 0), 2) # bounding rectangle (green for each detected shape)
            cv2.putText(imgContour, objType, (x + (w//2) - 10 , y + (h//2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


path = '/home/sr42/Projects/scaling-spoon/day2/taskResources/shapes.JPG'
img = cv2.imread(path)
imgContour = img.copy()

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7,7), 1)
imgCanny = cv2.Canny(imgBlur, 50, 50)
getContours(imgCanny)

cv2.imshow('Original', img)
cv2.imshow('Gray', imgGray)
cv2.imshow('Blur', imgBlur)
cv2.imshow('Canny', imgCanny)
cv2.imshow('Copy with contours', imgContour)

cv2.waitKey(0)