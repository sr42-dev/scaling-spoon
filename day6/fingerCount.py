import cv2
import time
import mediapipe as mp

class handDetector():

    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:

            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []

        if self.results.multi_hand_landmarks:

            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):

                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 15, (33, 32, 196), cv2.FILLED)

        return lmList


wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]
rating = 0
rating1 = 0
rating2 = 0
i = 0

while True:

    rating2 = rating1
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        fingers = []

        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:

            fingers.append(1)

        else:

            fingers.append(0)

        for id in range(1, 5):

            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:

                fingers.append(1)

            else:

                fingers.append(0)

        totalFingers = fingers.count(1)

        # print(totalFingers)

        if totalFingers == 0:
            h = '0'
            rating1 = 0
            cv2.putText(img, h, (100, 250), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        if totalFingers == 1:
            h = '1'
            rating1 = 1
            cv2.putText(img, h, (100, 250), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        if totalFingers == 2:
            h = '2'
            rating1 = 2
            cv2.putText(img, h, (100, 250), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        if totalFingers == 3:
            h = '3'
            rating1 = 3
            cv2.putText(img, h, (100, 250), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        if totalFingers == 4:
            h = '4'
            rating1 = 4
            cv2.putText(img, h, (100, 250), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        if totalFingers == 5:
            h = '5'
            rating1 = 5
            cv2.putText(img, h, (100, 250), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.rectangle(img, (50, 200), (175, 270), (0, 255, 0), 2)
    cv2.putText(img, 'Finger count (Backhand)', (50, 185), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
    cv2.imshow("Count", img)

    if rating1 == rating2:

        i += 1

        if i > 20:
            rating = rating1
            time.sleep(2)
            cv2.destroyAllWindows()
            cap.release()
            break

    elif rating1 != rating2:

        i = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break

