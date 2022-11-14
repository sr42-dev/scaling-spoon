# importing the necessary libraries
import os
import cv2
import time
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib.path as mplPath

# function to check if a point is present within a defined quadrilateral
def pointInQuad(x, y, x1, y1, x2, y2, x3, y3, x4, y4):

    polygon = mplPath.Path(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])) # drawing and storing the quadrilateral path
    point = (x, y)
    return polygon.contains_point(point) # checking for the point's existence within the drawn quadrilateral

# rotation matrix helper functions

# function to return the magnitude of a vector
def vec_length(v: np.array):
    return np.sqrt(sum(i ** 2 for i in v))

# function to process a vector parameter and return a normalized vector
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

# function to calculate and return a rotation matrix for quaternion generation
def look_at(eye: np.array, target: np.array):
    axis_z = normalize((eye - target))
    if vec_length(axis_z) == 0:
        axis_z = np.array((0, -1, 0))

    axis_x = np.cross(np.array((0, 0, 1)), axis_z)
    if vec_length(axis_x) == 0:
        axis_x = np.array((1, 0, 0))

    axis_y = np.cross(axis_z, axis_x)
    rot_matrix = np.matrix([axis_x, axis_y, axis_z]).transpose()
    return rot_matrix


# filter(s)

# Kalman filter in one dimension implemented as a class
class Kalman:

    def __init__(self, windowSize=10, n=5):
        # x: predicted angle
        # p: predicted angle uncertainty
        # n: number of iterations to run the filter for
        # dt: time interval for updates
        # q: process noise variance (uncertainty in the system's dynamic model)
        # r: measurement uncertainty
        # Z: list of position estimates derived from sensor measurements

        # initializing with static values due to very low variance in testing
        self.x = 0
        self.p = 0.5
        self.windowSize = windowSize
        self.n = n # must be smaller than windowSize
        self.Z = []

        self.q = 0 # assuming dynamic model uncertainty to be 0 (perfect system)
        self.dt = 0.05 # average latency is 50ms
        self.r = 0.5 # angle measurement uncertainty (determine experimentally based on test case)

    # prediction stage
    def predict(self):
        # prediction assuming a dynamic model
        self.x = self.x   # state transition equation
        self.p = self.p + self.q  # predicted covariance equation

    # measurement stage
    def measure(self, z):

        if len(self.Z) < self.windowSize:
            self.Z.append(z)
        else:
            self.Z.pop(0)
            self.Z.append(z)

        return np.mean(self.Z)

    # updation stage
    def update(self, z):
        k = self.p / (self.p + self.r)  # Kalman gain
        self.x = self.x + k * (z - self.x)  # state update
        self.p = (1 - k) * self.p  # covariance update

    # iterative processing stage
    def process(self, i):

        for j in range(1, self.n):
            self.predict()
            z = self.measure(i)
            self.update(z)

        return self.x

# streaming moving average filter in one dimension implemented as a class
class StreamingMovingAverage:

    def __init__(self, window_size):
        self.window_size = window_size # size of the window of values
        self.values = [] # list to hold said window
        self.sum = 0 # initializing the sum for the moving average

    # processing the average
    def process(self, value):
        self.values.append(value)
        self.sum += value
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
        return float(self.sum) / len(self.values)

# empty filter class implemented for comparative testing
class noFilter:

    def __init__(self):
        pass

    def process(self, value):
        return value

# pose detector class for mediapipe
class PoseDetector:

    """
    Estimates Pose points of a human body using the mediapipe library.
    """

    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    # a method to initialize the filters used in objects of this class (i.e.; single human obstacles detected by the mediapipe model)
    def filterSettings(self, angleFilter1, angleFilter2):

        self.angleFilter1 = angleFilter1
        self.angleFilter2 = angleFilter2

    # a method to detect and draw the landmarks detected by the model on the input frame
    def findPose(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    # a method to find the positions of the detected landmarks and return the same along with bounding box information
    def findPosition(self, img, draw=True, bboxWithHands=False):

        self.lmList = []
        self.bboxInfo = {}
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                self.lmList.append([id, cx, cy, cz])

            # bounding box generation
            ad = abs(self.lmList[12][1] - self.lmList[11][1]) // 2
            if bboxWithHands:
                x1 = self.lmList[16][1] - ad
                x2 = self.lmList[15][1] + ad
            else:
                x1 = self.lmList[12][1] - ad
                x2 = self.lmList[11][1] + ad

            y2 = self.lmList[29][2] + ad
            y1 = self.lmList[1][2] - ad
            bbox = (x1, y1, x2 - x1, y2 - y1)
            cx, cy = bbox[0] + (bbox[2] // 2), \
                     (bbox[1] + bbox[3] // 2) - 40

            self.bboxInfo = {"bbox": bbox, "center": (cx, cy)}

            if draw:
                cv2.rectangle(img, bbox, (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return self.lmList, self.bboxInfo

    # a method to extract the angle of orientation of the detected human obstacle by quaternion generation from the projection of two landmarks
    def generateQuaternion(self, p1, p2):

        if self.results.pose_landmarks != None:
            # calculating the rotation matrix
            orient = look_at(np.array([p1[1], p1[2], p1[3]]), np.array([p2[1], p2[2], p2[3]]))

            vec1 = np.array(orient[0], dtype=float)
            vec3 = np.array(orient[1], dtype=float)
            vec4 = np.array(orient[2], dtype=float)
            # normalize to unit length
            vec1 = vec1 / np.linalg.norm(vec1)
            vec3 = vec3 / np.linalg.norm(vec3)
            vec4 = vec4 / np.linalg.norm(vec4)

            M1 = np.zeros((3, 3), dtype=float)  # rotation matrix

            # rotation matrix setup
            M1[:, 0] = vec1
            M1[:, 1] = vec3
            M1[:, 2] = vec4

            # generating the quaternion
            r = np.math.sqrt(np.math.sqrt((float(1) + M1[0, 0] + M1[1, 1] + M1[2, 2]) ** 2)) * 0.5
            i = (M1[2, 1] - M1[1, 2]) / (4 * r)
            j = (M1[0, 2] - M1[2, 0]) / (4 * r)
            k = (M1[1, 0] - M1[0, 1]) / (4 * r)

            # converting quaternion to polar form
            A = np.math.sqrt((r ** 2) + (i ** 2) + (j ** 2) + (k ** 2))
            theta = np.math.acos(r / A)
            # B = np.math.sqrt((A ** 2) - (a ** 2))
            # cosphi1 = b1 / B
            # cosphi2 = b2 / B
            # cosphi3 = b3 / B

            realAngle = ((np.rad2deg(theta) / 45) - 1) * 180

            # filtering the reading
            realAngle1 = self.angleFilter1.process(realAngle)
            realAngleUnfiltered = self.angleFilter2.process(realAngle)

            return r, i, j, k, realAngle1, realAngleUnfiltered


def main():

    # initializing the frame and overlay settings
    cap = cv2.VideoCapture(0) # initializing the test video path
    cap.set(3, 720) # width
    cap.set(4, 480) # height

    # FPS initializations
    curTime = time.time() # initializing the starting time
    lastTime = curTime # initializing the time in the last frame to the current time itself
    fps = 0 # initializing the frame rate variable
    frameNumber = 0 # initializing the frame number to zero

    # pose detector settings and variables that visibly impact output (structured in this manner for ease of testing)
    detector = PoseDetector()
    # setting the filter options for the pose detector class
    # filter options : StreamingMovingAverage(10), Kalman(windowSize=20, n=10), noFilter()
    detector.filterSettings(angleFilter1=noFilter(), angleFilter2=Kalman(windowSize=20, n=10))
    drawState = True # the boolean determining whether or not landmarks and their associated line segments are to be drawn on the frame image

    # initializing data collection lists
    angle = []
    angleunfiltered = []
    times = []

    while True:

        # reading the image
        success, img = cap.read()

        # resizing the image to fit the frame
        img = cv2.resize(img, (768, 432))
        # flipping the image to get a real depiction of the scene
        img = cv2.flip(img, 1)

        # finding the landmarks and visualizing them
        img = detector.findPose(img, draw=drawState)

        # getting a list of landmarks and bounding box information
        lmList, bboxInfo = detector.findPosition(img, draw=False, bboxWithHands=False)

        # code to be executed if a human obstacle is detected (i.e.; if ia bounding box is generatable)
        if bboxInfo:

            # finding the center of the target pose
            center = bboxInfo["center"]
            yLocations = []
            for lm in lmList:
                yLocations.append(lm[2])

                # getting the landmarks corresponding to the shoulders
                if (lm[0] == 12):
                    lmrs = lm
                elif (lm[0] == 11):
                    lmls = lm

            # calculating the quaternion
            r, i, j, k, realAngleUnfiltered, realAngle = detector.generateQuaternion(lmrs, lmls)

            # appending the data to the data collection lists
            angle.append(realAngle)
            angleunfiltered.append(realAngleUnfiltered)
            times.append(time.time())

            # printing angle of approach
            cv2.putText(img, "Q = {0:.2f} + ".format(r) + "{0:.2f}i + ".format(i) + "{0:.2f}j + ".format(j) + "{0:.2f}k".format(k), (250, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img, '{0:.2f}'.format(realAngle), (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)
        
            # highlighting the shoulder points being considered for orientation evaluation
            cv2.circle(img, (lmls[1], lmls[2]), 5, (255, 255, 255), cv2.FILLED)
            cv2.circle(img, (lmrs[1], lmrs[2]), 5, (255, 255, 255), cv2.FILLED)
            cv2.line(img, (lmls[1], lmls[2]), (lmrs[1], lmrs[2]), (100, 255, 0), 1)


        # FPS calculation
        fps = 1 / (time.time() - curTime)
        curTime = time.time()
        cv2.putText(img, '{0:.2f}'.format(fps), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, '{0:.2f}'.format((1 / fps) * 1000), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)

        # scale
        cv2.putText(img, '50px: ', (675, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 0), 1, cv2.LINE_AA)
        cv2.line(img, (710, 17), (760, 17), (100, 255, 0), 1)
        cv2.line(img, (710, 17), (710, 22), (100, 255, 0), 1)
        cv2.line(img, (760, 17), (760, 22), (100, 255, 0), 1)

        # showing the processed frame
        cv2.imshow('Frame', img)

        # adding the video break conditions
        if (cv2.waitKey(1) == ord('q')) or (not success):
            cv2.imwrite('out.png', img)
            break

    # releasing & destroying the windows
    cap.release()
    cv2.destroyAllWindows()

    # plotting graphs & collecting data
    plt.plot(times, angle, label='realAngle 1-D Kalman filtered')
    plt.plot(times, angleunfiltered, label='realAngle Unfiltered')
    plt.title('realAngle measurements over time (Filtered vs Unfiltered)')
    plt.ylabel('Angle measurement in degrees')
    plt.xlabel('Time')
    plt.legend()
    plt.show()

# a definition of the main parameters
if __name__ == "__main__":

    main()