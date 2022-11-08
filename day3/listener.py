import rospy
from std_msgs.msg import String

def chatter_callback(message):
    rospy.init_node('listener', anonymous = True)

    rospy.Subscriber("chatter", String, chatter_callback)

    print("I heard %s", message.data)

    rospy.spin()

if __name__ == '__main__':
    try:
        chatter_callback()
    except rospy.ROSInterruptException:
        pass


