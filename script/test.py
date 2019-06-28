#! /usr/bin/env python

import rospy
from visualization_msgs.msg import Marker
from std_srvs.srv import *


def get_pos_cb(marker):
    rospy.loginfo("postion.x: %s", marker.pose.position.x)
    rospy.loginfo("postion.x: %s", marker.pose.position.y)
    rospy.loginfo("postion.x: %s", marker.pose.position.z)
 
def done_cb(req):
        rospy.loginfo("Cleared all faces")
        return {}

def main():    
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('get_obj', anonymous=True)
    rospy.Subscriber("visualization_marker", Marker, get_pos_cb)

    server = rospy.Service('done', Empty, done_cb)
    print "Ready to receive done."
    rospy.spin()

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    main()