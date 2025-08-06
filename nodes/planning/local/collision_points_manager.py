#!/usr/bin/env python3

import rospy
import shapely
import math
import numpy as np
import threading
from ros_numpy import msgify
from autoware_mini.msg import Path, DetectedObjectArray
from sensor_msgs.msg import PointCloud2

from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from autoware_mini.msg import TrafficLightResultArray

DTYPE = np.dtype([
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('vx', np.float32),
    ('vy', np.float32),
    ('vz', np.float32),
    ('distance_to_stop', np.float32),
    ('deceleration_limit', np.float32),
    ('category', np.int32)
])

class CollisionPointsManager:

    def __init__(self):

        # parameters
        self.safety_box_width = rospy.get_param("safety_box_width")
        self.stopped_speed_limit = rospy.get_param("stopped_speed_limit")
        self.braking_safety_distance_obstacle = rospy.get_param("~braking_safety_distance_obstacle")
        self.braking_safety_distance_goal = rospy.get_param("~braking_safety_distance_goal")
        self.braking_safety_distance_stopline= rospy.get_param("~braking_safety_distance_stopline")
        # Parameters related to lanelet2 map loading
        coordinate_transformer = rospy.get_param("/localization/coordinate_transformer")
        use_custom_origin = rospy.get_param("/localization/use_custom_origin")
        utm_origin_lat = rospy.get_param("/localization/utm_origin_lat")
        utm_origin_lon = rospy.get_param("/localization/utm_origin_lon")
        lanelet2_map_path = rospy.get_param("~lanelet2_map_path")


        # variables
        self.detected_objects = None
        self.goal_waypoint = None
        self.trafficlights = None
        self.tfl_stoplines = None
        self.stopline_statuses = {}

        # Lock for thread safety
        self.lock = threading.Lock()

        # Load the map using Lanelet2
        if coordinate_transformer == "utm":
            projector = UtmProjector(Origin(utm_origin_lat, utm_origin_lon), use_custom_origin, False)
        else:
            raise RuntimeError('Only "utm" is supported for lanelet2 map loading')
        lanelet2_map = load(lanelet2_map_path, projector)

        all_stoplines = get_stoplines(lanelet2_map)
        self.trafficlights = get_stoplines_trafficlights(lanelet2_map)
        # If stopline_id is not in self.signals then it has no signals (traffic lights)
        self.tfl_stoplines = {k: v for k, v in all_stoplines.items() if k in self.trafficlights}

        # publishers
        self.local_path_collision_pub = rospy.Publisher('collision_points', PointCloud2, queue_size=1, tcp_nodelay=True)

        # subscribers
        rospy.Subscriber('extracted_local_path', Path, self.path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('global_path', Path, self.global_path_callback, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/detection/final_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=1, buff_size=2**20, tcp_nodelay=True)
        #subscriber for traffic light status
        rospy.Subscriber('/detection/traffic_light_status', TrafficLightResultArray, self.traffic_light_status_callback, queue_size=1, tcp_nodelay=True)
    
    def traffic_light_status_callback(self, msg):
        stopline_statuses = {}
        for result in msg.results:
            stopline_statuses[result.stopline_id] = result.recognition_result

        self.stopline_statuses = stopline_statuses
    
    def global_path_callback(self, msg):
        if len(msg.waypoints) == 0:
            rospy.logwarn_throttle(5, "%s - Received global path with no waypoints!", rospy.get_name())
            return

        # extract the last waypoint as the goal
        with self.lock:
            self.goal_waypoint = msg.waypoints[-1]
            rospy.loginfo_throttle(5, "%s - Updated goal waypoint at (%.2f, %.2f)", rospy.get_name(),
                                self.goal_waypoint.position.x, self.goal_waypoint.position.y)

    
    def detected_objects_callback(self, msg):
        self.detected_objects = msg.objects

    def path_callback(self, msg):
        with self.lock:
            detected_objects = self.detected_objects
        collision_points = np.array([], dtype=DTYPE)

        stopline_statuses = self.stopline_statuses

        if len(msg.waypoints) > 0:

            local_path_linestring = shapely.LineString([(waypoint.position.x, waypoint.position.y) for waypoint in msg.waypoints])
            local_path_buffer = local_path_linestring.buffer(self.safety_box_width / 2, cap_style="flat")
            shapely.prepare(local_path_buffer)

            if len(detected_objects) > 0:

                for obj in detected_objects:
                    convex_hull_array = np.array(obj.convex_hull).reshape(-1, 3)
                    convex_hull_2d = convex_hull_array[:, :2]
    
                    object_polygon = shapely.Polygon(convex_hull_2d)

                
                    if not object_polygon.intersects(local_path_buffer):
                        continue

                    intersection_geom = object_polygon.intersection(local_path_buffer)
                    intersection_coords = shapely.get_coordinates(intersection_geom)

                    object_speed = np.linalg.norm([
                        obj.velocity.x,
                        obj.velocity.y,
                        obj.velocity.z
                    ])

                    for x, y in intersection_coords:
                        collision_points = np.append(collision_points, np.array([(x, y, obj.centroid.z, obj.velocity.x, obj.velocity.y, obj.velocity.z,
                                                                                                    self.braking_safety_distance_obstacle, np.inf, 3 if object_speed < self.stopped_speed_limit else 4)], dtype=DTYPE))
            if len(stopline_statuses) > 0:
                for stopline_id, stopline_linestring in self.tfl_stoplines.items():
                    # CLASSIFIER_RESULT_TO_TLRESULT is 0 for red and yellow, so with stopline_statuses[stopline_id] 
                    # we retrieve recognition_result with this if it is equal to 0, we need to stop
                    if stopline_id in stopline_statuses and stopline_statuses[stopline_id] == 0 and stopline_linestring.intersects(local_path_linestring):
                        intersection_point = local_path_linestring.intersection(stopline_linestring)
                        collision_points = np.append(collision_points, np.array([(intersection_point.x, intersection_point.y, 0,
                                                                            0.0, 0.0, 0.0,  # no velocity because its a static point
                                                                            self.braking_safety_distance_stopline, np.inf,
                                                                            2)], 
                                                                            dtype=DTYPE))

            
            
            # goal waypoint processing
            if self.goal_waypoint is not None:
                goal_point = shapely.Point(self.goal_waypoint.position.x, self.goal_waypoint.position.y)
                goal_buffer = goal_point.buffer(0.5) # 0.5 expands the goal point slightly (into a small circular area) and gives tolerance to account for small path/goal mismatches
                if goal_buffer.intersects(local_path_buffer):
                    collision_points = np.append(collision_points, np.array([(goal_point.x, goal_point.y, self.goal_waypoint.position.z,
                                                                            0.0, 0.0, 0.0,  # no velocity because its a static point
                                                                            self.braking_safety_distance_goal, np.inf,
                                                                            1)],  # category 1 = goal
                                                                            dtype=DTYPE))

            
            pointcloud_msg = msgify(PointCloud2, collision_points)
            pointcloud_msg.header = msg.header
            self.local_path_collision_pub.publish(pointcloud_msg)
            return

        empty_message = PointCloud2()
        empty_message.header = msg.header
        self.local_path_collision_pub.publish(empty_message)

    def run(self):
        rospy.spin()

def get_stoplines(lanelet2_map):
    """
    Add all stop lines to a dictionary with stop_line id as key and stop_line as value
    :param lanelet2_map: lanelet2 map
    :return: {stop_line_id: stopline, ...}
    """

    stoplines = {}
    for line in lanelet2_map.lineStringLayer:
        if line.attributes:
            if line.attributes["type"] == "stop_line":
                # add stoline to dictionary and convert it to shapely LineString
                stoplines[line.id] = shapely.LineString([(p.x, p.y) for p in line])

    return stoplines


def get_stoplines_trafficlights(lanelet2_map):
    """
    Iterate over all regulatory_elements with subtype traffic light and extract the stoplines and sinals.
    Organize the data into dictionary indexed by stopline id that contains a traffic_light id and the four coners of the traffic light.
    :param lanelet2_map: lanelet2 map
    :return: {stopline_id: {traffic_light_id: {'top_left': [x, y, z], 'top_right': [...], 'bottom_left': [...], 'bottom_right': [...]}, ...}, ...}
    """

    signals = {}

    for reg_el in lanelet2_map.regulatoryElementLayer:
        if reg_el.attributes["subtype"] == "traffic_light":
            # ref_line is the stop line and there is only 1 stopline per traffic light reg_el
            linkId = reg_el.parameters["ref_line"][0].id

            for tfl in reg_el.parameters["refers"]:
                tfl_height = float(tfl.attributes["height"])
                # plId represents the traffic light (pole), one stop line can be associated with multiple traffic lights
                plId = tfl.id

                traffic_light_data = {'top_left': [tfl[0].x, tfl[0].y, tfl[0].z + tfl_height],
                                      'top_right': [tfl[1].x, tfl[1].y, tfl[1].z + tfl_height],
                                      'bottom_left': [tfl[0].x, tfl[0].y, tfl[0].z],
                                      'bottom_right': [tfl[1].x, tfl[1].y, tfl[1].z]}

                # signals is a dictionary indexed by stopline id and contains dictionary of traffic lights indexed by pole id
                # which in turn contains a dictionary of traffic light corners
                signals.setdefault(linkId, {}).setdefault(plId, traffic_light_data)

    return signals

if __name__ == '__main__':
    rospy.init_node('collision_points_manager')
    node = CollisionPointsManager()
    node.run()