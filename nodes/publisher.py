#!/usr/bin/env python3
import rospy
from std_msgs.msg import String

rospy.init_node('publisher')
hz_rate = message = rospy.get_param('~publish_frequency', 10)
rate = rospy.Rate(hz_rate)
pub = rospy.Publisher('/message', String, queue_size=10)
message = rospy.get_param('~message', 'Hello World!')

while not rospy.is_shutdown():
    pub.publish(message)
    rate.sleep()