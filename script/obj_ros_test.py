#! /usr/bin/env python
from collections import OrderedDict
import numpy as np
from sawyer_control.envs.sawyer_env_base import SawyerEnvBase
from sawyer_control.core.serializable import Serializable
from sawyer_control.core.eval_util import get_stat_in_paths, \
    create_stats_ordered_dict

import rospy
from visualization_msgs.msg import Marker
from std_srvs.srv import *


def target_pos_cb(marker):
    rospy.loginfo("postion.x: %s", marker.pose.position.x)
    rospy.loginfo("postion.y: %s", marker.pose.position.y)
    rospy.loginfo("postion.z: %s", marker.pose.position.z)
 
def done_cb(req):
        rospy.loginfo("done")
        return {}

def main():    
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('get_obj', anonymous=True)
    rospy.Subscriber("visualization_marker", Marker, target_pos_cb)

    server = rospy.Service('done', Empty, done_cb)
    rospy.loginfo("Ready to receive done")
    rospy.spin()

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    main()