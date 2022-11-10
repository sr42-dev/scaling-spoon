#!/usr/bin/env python3
# this node moves the quadrotor in a pre-planned path

import math
import rospy
from time import sleep
from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped


rospy.init_node('autohector', anonymous=False)
vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)


def initMessage(vx,vy,vz,vaz):
    vel_msg = Twist()
    vel_msg.linear.x = float(vx)
    vel_msg.linear.y = float(vy)
    vel_msg.linear.z = float(vz)
    vel_msg.angular.z = float(vaz)
    vel_msg.angular.x = float(0.0)
    vel_msg.angular.y = float(0.0)
    vel_pub.publish(vel_msg)

def hover():
    initMessage(0.0,0.0,0.0,0.0)

def up():
    vel_msg = Twist()
    vel_msg.linear.z = float(1.0)
    vel_pub.publish(vel_msg)

def down():
    vel_msg = Twist()
    vel_msg.linear.z = float(-1.0)
    vel_pub.publish(vel_msg)

def forward():
    vel_msg = Twist()
    vel_msg.linear.x = float(1.0)
    vel_pub.publish(vel_msg)

def backward():
    vel_msg = Twist()
    vel_msg.linear.x = float(-1.0)
    vel_pub.publish(vel_msg)

def right():
    vel_msg = Twist()
    vel_msg.linear.y = float(-1.0)
    vel_pub.publish(vel_msg)

def left():
    vel_msg = Twist()
    vel_msg.linear.y = float(1.0)
    vel_pub.publish(vel_msg)

def cw():
    vel_msg = Twist()
    vel_msg.angular.z = float(-1.0)
    vel_pub.publish(vel_msg)

def ccw():
    vel_msg = Twist()
    vel_msg.angular.z = float(1.0)
    vel_pub.publish(vel_msg)

def pathExec():

    # write the movements to be executed by the drone sequentially
    # note that the last provided command is what will be published

    # takeoff   
    hover()
    sleep(2)
    up()
    sleep(3)
    hover()
    sleep(2)

    forward()
    sleep(1)
    hover()
    sleep(2)

    right()
    sleep(1)
    hover()
    sleep(2)

    forward()
    sleep(5)
    hover()
    sleep(2)

    cw()
    sleep(4)
    hover()
    sleep(2)

    forward()
    sleep(1)
    hover()
    sleep(2)

    cw()
    sleep(4)
    hover()
    sleep(2)

    forward()
    sleep(5)
    hover()
    sleep(2)

    ccw()
    sleep(4)
    hover()
    sleep(2)

    forward()
    sleep(1)
    hover()
    sleep(2)

    ccw()
    sleep(4)
    hover()
    sleep(2)

    forward()
    sleep(3)
    hover()
    sleep(2)

    forward()
    sleep(1)
    hover()
    sleep(2)

    right()
    sleep(1)
    hover()
    sleep(2)

    forward()
    sleep(5)
    hover()
    sleep(2)

    cw()
    sleep(4)
    hover()
    sleep(2)

    forward()
    sleep(1)
    hover()
    sleep(2)

    cw()
    sleep(4)
    hover()
    sleep(2)

    forward()
    sleep(5)
    hover()
    sleep(2)

    ccw()
    sleep(4)
    hover()
    sleep(2)

    forward()
    sleep(1)
    hover()
    sleep(2)

    ccw()
    sleep(4)
    hover()
    sleep(2)

    forward()
    sleep(3)
    hover()
    sleep(2)


if __name__ == '__main__':

    pathExec()
