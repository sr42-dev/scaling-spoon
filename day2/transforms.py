# HSV limit finding from webcam feed

import cv2
import numpy as np


def empty(a):  # argument required
    pass

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

cap = cv2.VideoCapture(0) # 0 - default webcam
cap.set(3, 640) # width
cap.set(4, 480) # height
cap.set(10, 100) # brightness

cv2.namedWindow('Trackbars')  # Creating trackbars to isolate required color
cv2.resizeWindow('Trackbars', 640, 240)

# cv2.createTrackbar('H minimum', 'Trackbars', 0, 179, empty) # 180 hues available in opencv (lower and upper limits for trackbars), empty is a function called each time the trackbar is changed
# cv2.createTrackbar('H maximum', 'Trackbars', 179, 179, empty) # initial trackbars for color detection and limit identification
# cv2.createTrackbar('S minimum', 'Trackbars', 0, 255, empty)
# cv2.createTrackbar('S maximum', 'Trackbars', 255, 255, empty)
# cv2.createTrackbar('V minimum', 'Trackbars', 0, 255, empty)
# cv2.createTrackbar('V maximum', 'Trackbars', 255, 255, empty)

cv2.createTrackbar('H minimum', 'Trackbars', 29, 179, empty)  # trackbars for specific colour
cv2.createTrackbar('H maximum', 'Trackbars', 146, 179, empty)
cv2.createTrackbar('S minimum', 'Trackbars', 13, 255, empty)
cv2.createTrackbar('S maximum', 'Trackbars', 93, 255, empty)
cv2.createTrackbar('V minimum', 'Trackbars', 66, 255, empty)
cv2.createTrackbar('V maximum', 'Trackbars', 127, 255, empty)

while True:

    success, img = cap.read() # <successful execution (boolean)>, <image variable>

    hMin = cv2.getTrackbarPos('H minimum', 'Trackbars')
    hMax = cv2.getTrackbarPos('H maximum', 'Trackbars')
    sMin = cv2.getTrackbarPos('S minimum', 'Trackbars')
    sMax = cv2.getTrackbarPos('S maximum', 'Trackbars')
    vMin = cv2.getTrackbarPos('V minimum', 'Trackbars')
    vMax = cv2.getTrackbarPos('V maximum', 'Trackbars')
    # print(hMin, hMax, sMin, sMax, vMin, vMax)

    # color filtering 
    #imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # conversion to HSV from BGR
    #lower = np.array([hMin, sMin, vMin])  # minimum range array
    #upper = np.array([hMax, sMax, vMax])  # maximum range array
    #mask = cv2.inRange(imgHSV, lower, upper)  # filtering out colours from HSV image


    # dilation 
    #imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #imgGrayBlur = cv2.GaussianBlur(imgGray, (7,7), 0)
    #imgCanny = cv2.Canny(img, 100, 100)
    #imgDilation = cv2.dilate(imgCanny, (100,100), iterations=1) # <Canny image variable>, <kernel (matrix)>, iterations = <thickness>


    # erosion
    #imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #imgGrayBlur = cv2.GaussianBlur(imgGray, (7,7), 0)
    #imgCanny = cv2.Canny(img, 100, 100)
    #imgEroded = cv2.erode(imgCanny, (10,10), iterations=1) # similar parameters to dilate()   


    # blurred grayscale 
    #imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #imgGrayBlur = cv2.GaussianBlur(imgGray, (7,7), 0) # <image variable>, <kernal size (odd square)>, <sigma x>


    # cropping
    imgCropped = img[0:200, 200:500] # height first, then width


    # drawing shapes
    #cv2.line(img,(0,0), (img.shape[1],img.shape[0]), (0, 255, 255), 3) # <image variable>, <start coordinates>, <end coordinates>, <colour>, <thickness>#
    #cv2.rectangle(img, (0,0), (450,450), (0,0,255), 2) # image, diagonal start, diagonal end, color, thickness
    #cv2.rectangle(img, (0,0), (250,250), (0,0,255), cv2.FILLED) # cv2.FILLED fills color of boundary in defined rectangle
    #cv2.circle(img, (450,50), 30, (255,255,0), 5) # base image, centre coordinates, color, thickness
    #cv2.circle(img, (450,200), 30, (255,255,0), cv2.FILLED) # base image, centre coordinates, color, fill status
    #cv2.putText(img, "OpenCV", (300,200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 1) # base image, text, origin coordinates, font, scale(size, decimals allowed), color, thickness
    #cv2.putText(img, "OpenCV", (300,300), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 150, 0), 1)
    #cv2.putText(img, "OpenCV", (300,400), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 5)


    # resizing
    imgResize = cv2.resize(img, (300,200)) # <image variable>, <dimensions (width, height)>


    # result display (replace 'None with your image variable')
    imgResult = imgResize  # adds two images and creates a new one where non black colours on the mask are given colour from the original
    imgStacked = stackImages(0.5, ([img, imgResult])) # stacking images
    cv2.imshow('Test window', imgResize) # showing image
    cv2.imshow('Test window', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print()
print('Required values : ')
print('hMin, sMin, vMin, hMax, sMax, vMax = ', hMin, ',', sMin, ',', vMin, ',', hMax, ',', sMax, ',', vMax)