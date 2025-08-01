#!/usr/bin/env python3

import math
import rospy

# All these imports from lanelet2 library should be sufficient
import lanelet2
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findNearest

# Other imports required for the lanelet2_global_planner
from geometry_msgs.msg import PoseStamped
from autoware_mini.msg import Path, Waypoint
from shapely import distance
from shapely.geometry import Point, LineString



def load_lanelet2_map(lanelet2_map_path, coordinate_transformer, use_custom_origin, utm_origin_lat, utm_origin_lon):
    """
    Load a lanelet2 map from a file and return it
    :param lanelet2_map_path: name of the lanelet2 map file
    :param coordinate_transformer: coordinate transformer
    :param use_custom_origin: use custom origin
    :param utm_origin_lat: utm origin latitude
    :param utm_origin_lon: utm origin longitude
    :return: lanelet2 map
    """


    # Load the map using Lanelet2
    if coordinate_transformer == "utm":
        projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), use_custom_origin, False)
    else:
        raise ValueError('Unknown coordinate_transformer for loading the Lanelet2 map ("utm" should be used): ' + coordinate_transformer)

    lanelet2_map = load(lanelet2_map_path, projector)

    return lanelet2_map


class GlobalPlanner:
    def __init__(self):
        # Parameters
        self.lanelet2_map_path = rospy.get_param("~lanelet2_map_path")
        self.speed_limit = rospy.get_param("~speed_limit")
        self.output_frame = rospy.get_param("/planning/lanelet2_global_planner/output_frame")
        self.coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        self.use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        self.utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        self.utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")
        self.distance_to_goal_limit = rospy.get_param("/planning/lanelet2_global_planner/distance_to_goal_limit")
        self.lanelet2_map = load_lanelet2_map(self.lanelet2_map_path,self.coordinate_transformer,self.use_custom_origin, self.utm_origin_lat,self.utm_origin_lon)
        self.current_location = None
        self.goal_point = None
        traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                lanelet2.traffic_rules.Participants.VehicleTaxi)
        self.graph = lanelet2.routing.RoutingGraph(self.lanelet2_map, traffic_rules)

        # Publishers
        self.waypoints_pub = rospy.Publisher('global_path', Path, queue_size=10, latch=True)

        # Subscribers
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback, queue_size=1)
        rospy.Subscriber('/localization/current_pose', PoseStamped, self.current_pose_callback, queue_size=1)

    def goal_callback(self, msg):
        self.goal_point = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)

        # loginfo message about receiving the goal point
        rospy.loginfo("%s - goal position (%f, %f, %f) orientation (%f, %f, %f, %f) in %s frame", rospy.get_name(),
                    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z,
                    msg.pose.orientation.w, msg.header.frame_id)
        
        path = self.compute_path()
        if path is not None:
            waypoints = self.from_lanelet2_to_waypoints(path)
            self.waypoints_publishing(waypoints)
    
    def current_pose_callback(self, msg):
        self.current_location = BasicPoint2d(msg.pose.position.x, msg.pose.position.y)

        if self.goal_point is not None:
            d = math.sqrt((self.current_location.x - self.goal_point.x)**2+(self.current_location.y - self.goal_point.y)**2)
            if d <=  self.distance_to_goal_limit:
                self.waypoints_publishing([])
                rospy.loginfo("%s - goal reached, path has been cleared!", rospy.get_name())
                self.goal_point = None


    def compute_path(self):
        # get start and end lanelets
        start_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.current_location, 1)[0][1]
        goal_lanelet = findNearest(self.lanelet2_map.laneletLayer, self.goal_point, 1)[0][1]
        
        # find routing graph
        route = self.graph.getRoute(start_lanelet, goal_lanelet, 0, True)

        if route is None:
            rospy.logwarn("No path has been found after entering the goal point")
            return None
        else:
            # find shortest path
            path = route.shortestPath()
            path_no_lane_change = path.getRemainingLane(start_lanelet)
            return path_no_lane_change

    def from_lanelet2_to_waypoints(self, path):
        waypoints = []
        coords = []

        last_lanelet = False

        for i, lanelet in enumerate(path):
            if i == len(path)-1:
                last_lanelet = True

            if 'speed_ref' in lanelet.attributes:
                speed = float(lanelet.attributes['speed_ref'])
                if speed > self.speed_limit:
                    speed = self.speed_limit
            else:
                speed = self.speed_limit

            for idx, point in enumerate(lanelet.centerline):
                if not last_lanelet and idx == len(lanelet.centerline)-1:
                    break

                coords.append((point.x, point.y, point.z))  # save all points in a list to build linestring later

                waypoint = Waypoint()
                waypoint.position.x = point.x
                waypoint.position.y = point.y
                waypoint.position.z = point.z
                waypoint.speed = speed / 3.6
                waypoints.append(waypoint)

        # post-processing to align last point to goal
        if self.goal_point and len(coords) >= 2:

            line = LineString([(x, y) for x, y, z in coords])
            goal_shapely = Point(self.goal_point.x, self.goal_point.y)

            proj_dist = line.project(goal_shapely)
            proj_point = line.interpolate(proj_dist)

            final_wp = Waypoint()
            final_wp.position.x = proj_point.x
            final_wp.position.y = proj_point.y
            final_wp.position.z = waypoints[-1].position.z  
            final_wp.speed = waypoints[-1].speed

            waypoints[-1] = final_wp  
            
        return waypoints

    
    def waypoints_publishing(self, waypoints):
        path = Path()        
        path.header.frame_id = self.output_frame
        path.header.stamp = rospy.Time.now()
        path.waypoints = waypoints
        self.waypoints_pub.publish(path)


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('lanelet2_global_planner')
    node = GlobalPlanner()
    node.run()