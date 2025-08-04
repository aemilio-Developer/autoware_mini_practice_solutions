#!/usr/bin/env python3

import math
import rospy

from sensor_msgs.msg import PointCloud2
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
from ros_numpy import numpify, msgify
import numpy as np
from sklearn.cluster import DBSCAN

class PointsClusterer:
    def __init__(self):

        # Parameters
        self.cluster_epsilon = rospy.get_param("~cluster_epsilon")
        self.cluster_min_size = rospy.get_param("~cluster_min_size")
        self.clusterer = DBSCAN(eps=self.cluster_epsilon, min_samples=self.cluster_min_size)

        # Publisher
        self.cluster_pub = rospy.Publisher('points_clustered', PointCloud2, queue_size=1, tcp_nodelay=True)

        # Subscriber
        rospy.Subscriber('points_filtered', PointCloud2, self.points_callback, queue_size=1, buff_size=2**24, tcp_nodelay=True)


    def points_callback(self, msg):
        data = numpify(msg)
        points = structured_to_unstructured(data[['x', 'y', 'z']], dtype=np.float32)
        labels = self.clusterer.fit_predict(points)

        assert points.shape[0] == labels.shape[0], (
        f"Number of points ({points.shape[0]}) does not match number of labels ({labels.shape[0]})"
        )

        #labels.reshape(-1, 1) ensures it's column-shaped for hstack.
        #np.hstack((points, labels)) joins (N, 3) with (N, 1) → (N, 4).
        #filtering labels[:, 0] != -1 drops noise points (those labeled -1).

        labels = labels.reshape(-1, 1)  # From (N,) to (N, 1)
        points_with_labels = np.hstack((points, labels))  # [x, y, z, label]
        points_labeled = points_with_labels[labels[:, 0] != -1]

        # convert labelled points to PointCloud2 format
        data = unstructured_to_structured(points_labeled, dtype=np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('label', np.int32)
        ]))

        # publish clustered points message
        cluster_msg = msgify(PointCloud2, data)
        cluster_msg.header.stamp = msg.header.stamp
        cluster_msg.header.frame_id = msg.header.frame_id
        self.cluster_pub.publish(cluster_msg)


    def run(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('points_clusterer')
    node = PointsClusterer()
    node.run()
