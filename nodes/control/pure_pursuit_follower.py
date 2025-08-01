#!/usr/bin/env python3

import rospy
import math

from autoware_mini.msg import Path
from autoware_mini.msg import VehicleCmd
from geometry_msgs.msg import PoseStamped
from shapely.geometry import LineString, Point
from shapely import prepare, distance
from tf.transformations import euler_from_quaternion
import numpy as np
from scipy.interpolate import interp1d

class PurePursuitFollower:
    def __init__(self):

        # Parameters
        self.path_linstring = None
        self.lookahead_distance = rospy.get_param("~lookahead_distance")
        self.wheel_base = rospy.get_param("/vehicle/wheel_base")
        self.distance_to_velocity_interpolator = None
        self.end_of_track = False

        # Publishers
        self.vehicle_cmd_pub = rospy.Publisher('/control/vehicle_cmd', VehicleCmd, queue_size=10)

        # Subscribers
        rospy.Subscriber('path', Path, self.path_callback, queue_size=1)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1)

    def path_callback(self, msg):

        # convert waypoints to shapely linestring
        self.path_linstring = LineString([(w.position.x, w.position.y) for w in msg.waypoints])
        # prepare path - creates spatial tree, making the spatial queries more efficient
        prepare(self.path_linstring)

        if msg.waypoints:
            self.end_of_track = False
            # collect waypoint x and y coordinates
            waypoints_xy = np.array([(w.position.x, w.position.y) for w in msg.waypoints])
            # Calculate distances between points
            distances = np.cumsum(np.sqrt(np.sum(np.diff(waypoints_xy, axis=0)**2, axis=1)))
            # add 0 distance in the beginning
            distances = np.insert(distances, 0, 0)
            # Extract velocity values at waypoints
            velocities = np.array([w.speed for w in msg.waypoints])
            # Create a distance-to-velocity interpolator for the path
            self.distance_to_velocity_interpolator = interp1d(distances, velocities, kind='linear', fill_value=0.0, bounds_error=False)
        else:
            self.end_of_track = True
        

    def current_pose_callback(self, msg):

            current_pose = Point([msg.pose.position.x, msg.pose.position.y])
            velocity = 0.0
            steering_angle = 0.0

            if self.end_of_track is False and self.path_linstring is not None:
                d_ego_from_path_start = self.path_linstring.project(current_pose)

                _, _, heading = euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
                
                lookahead_position = d_ego_from_path_start + self.lookahead_distance
                lookahead_point = self.path_linstring.interpolate(lookahead_position)
                # lookahead point heading calculation
                lookahead_heading = np.arctan2(lookahead_point.y - current_pose.y, lookahead_point.x - current_pose.x)

                # Compute the actual Euclidean distance
                ld = distance(current_pose,lookahead_point) 
                #calculate the steering angle
                alpha = lookahead_heading - heading
                steering_angle = np.arctan((2*self.wheel_base*math.sin(alpha))/ld)

                if self.distance_to_velocity_interpolator is not None:
                    velocity = self.distance_to_velocity_interpolator(d_ego_from_path_start)
            
            vehicle_cmd = VehicleCmd()
            vehicle_cmd.header.stamp = msg.header.stamp
            vehicle_cmd.header.frame_id = "base_link"
            vehicle_cmd.ctrl_cmd.steering_angle = steering_angle
            vehicle_cmd.ctrl_cmd.linear_velocity = velocity
            self.vehicle_cmd_pub.publish(vehicle_cmd)


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('pure_pursuit_follower')
    node = PurePursuitFollower()
    node.run()