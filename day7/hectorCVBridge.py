# this node takes an input stream from the camera feed and records the ground location of the drone in areas with detected targets

import sys
import cv2
import rospy
import numpy as np
import pandas as pd
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()

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


def imageCallback(ros_image):

    global bridge
    global out

    try:
        img = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    except CvBridgeError as e:
            print(e)

    ##############################################################
    # from this point on, 'img' is the target of processing
    ##############################################################

    # image processing
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        
    imgStacked = stackImages(0.5, ([img, imgHSV]))
    cv2.imshow("Image", imgStacked)
    ##############################################################
    cv2.waitKey(1)

  
def main(args):

    global out

    rospy.init_node('UAVRecordRaw', anonymous=True)

    #`for turtlebot3 waffle
    #`image_topic="/camera/rgb/image_raw/compressed"
    #`for usb cam
    #`image_topic = "/usb_cam/image_raw"
    #`for hector quadrotor
    imageTopic = "/front_cam/camera/image"

    image_sub = rospy.Subscriber(imageTopic,Image, imageCallback)

    print("Press ctrl + c to trigger a KeyboardInterrupt.")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down.")

    out.release() 
    cv2.destroyAllWindows()
    print("Shutdown complete.")

if __name__ == '__main__':

    main(sys.argv)