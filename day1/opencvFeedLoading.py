# Reading images, video & webcam feeds

import cv2

print('package imported')

'''
# Display image

img = cv2.imread('resources/The Sci-Fi Archive.jpg')

cv2.imshow('output', img)
cv2.waitKey(0) # 0 - infinite, non zero - number of milliseconds
'''

# Play video

cap = cv2.VideoCapture('resources/gae_ofc.mp4')

while True :
    success, img = cap.read() # <successful execution (boolean)>, <image variable>
    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''

'''
# Real time webcam feed

cap = cv2.VideoCapture(0) # 0 - default webcam
cap.set(3, 640) # width
cap.set(4, 480) # height
cap.set(10, 100) # brightness

while True :
    success, img = cap.read() # <successful execution (boolean)>, <image variable>
    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

