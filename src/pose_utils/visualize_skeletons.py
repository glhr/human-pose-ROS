#! /usr/bin/env python3
import rospy
import numpy as np
from human_pose_ROS.msg import Skeleton, PoseEstimation
from rospy_message_converter import message_converter

from vision_utils.img import image_to_numpy, numpy_to_image, load_image
from vision_utils.logger import get_logger, get_printer
from vision_utils.timing import get_timestamp

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

logger=get_logger()
pp = get_printer()

cam = True
FRAME_ID = "/world"

connected_points = [
(0,2), (2,4), (4,6), (6,8), (8,10),
(0,1), (1,3), (3,5), (5,7), (7,9),
(5,11), (11,13), (13,15),
(6,12), (12,14), (14,16),
(11,12), (5,6)]

colors = dict()
for k in range(100):
  colors[k] = tuple(np.random.randint(256, size=3)/256)

import argparse
parser = argparse.ArgumentParser(description='Visualization options')
parser.add_argument('--lifetime',
                 default=1,
                 help='Marker lifetime')

args, unknown = parser.parse_known_args()

def pose_cb(msg):

    for n, skeleton in enumerate(msg.skeletons):
        skeleton_dict = message_converter.convert_ros_message_to_dictionary(skeleton)
        for i,conn in enumerate(connected_points):
            label_1 = list(skeleton_dict.keys())[conn[0]]
            label_2 = list(skeleton_dict.keys())[conn[1]]
            logger.debug(f"{label_1} --> {label_2}")
            pnt_1 = skeleton_dict[label_1]
            pnt_2 = skeleton_dict[label_2]
            now = rospy.get_rostime()

            skeleton_i = skeleton.id if skeleton.id else n

            line_marker = Marker()
            line_marker.header.frame_id = FRAME_ID
            line_marker.type = line_marker.LINE_STRIP
            line_marker.action = line_marker.ADD
            line_marker.scale.x = 0.02
            line_marker.color.a = 1.0
            line_marker.color.r, line_marker.color.g, line_marker.color.b = colors[skeleton_i]
            line_marker.pose.orientation.w = 1.0
            line_marker.pose.position.x, line_marker.pose.position.y, line_marker.pose.position.z = 0, 0, 0
            line_marker.id = (skeleton_i+1)*1000 + i*2+1
            line_marker.lifetime = rospy.Duration(float(args.lifetime))
            line_marker.header.stamp = now
            line_marker.points = []

            if len(pnt_1) and len(pnt_2):
                # first point
                first_line_point = Point()
                first_line_point.x, first_line_point.y, first_line_point.z = pnt_1
                line_marker.points.append(first_line_point)
                # second point
                second_line_point = Point()
                second_line_point.x, second_line_point.y, second_line_point.z = pnt_2
                line_marker.points.append(second_line_point)

                pp.pprint(line_marker.points)
                skel_pub.publish(line_marker)

            pnt_marker = Marker()
            pnt_marker.header.frame_id = FRAME_ID
            pnt_marker.type = pnt_marker.SPHERE
            pnt_marker.action = pnt_marker.ADD
            pnt_marker.scale.x, pnt_marker.scale.y, pnt_marker.scale.z = 0.03, 0.03, 0.03
            pnt_marker.color.a = 1.0
            pnt_marker.color.r, pnt_marker.color.g, pnt_marker.color.b = (1.0,1.0,1.0)
            pnt_marker.pose.orientation.w = 1.0
            pnt_marker.lifetime = rospy.Duration(float(args.lifetime))
            pnt_marker.header.stamp = now

            if len(pnt_1):
                pnt_marker.pose.position.x, pnt_marker.pose.position.y, pnt_marker.pose.position.z = pnt_1
                # logger.debug(pnt_marker.pose.position)
                pnt_marker.id = skeleton_i*100 + i*2
                skel_pub.publish(pnt_marker)
            if len(pnt_2):
                pnt_marker.pose.position.x, pnt_marker.pose.position.y, pnt_marker.pose.position.z = pnt_2
                pnt_marker.id = skeleton_i*100 + i*2+1
                skel_pub.publish(pnt_marker)

            skel_centroid = skeleton_dict['centroid']
            centroid_marker = Marker()
            centroid_marker.header.frame_id = FRAME_ID
            centroid_marker.type = centroid_marker.SPHERE
            centroid_marker.action = centroid_marker.ADD
            centroid_marker.scale.x, centroid_marker.scale.y, centroid_marker.scale.z = 0.1, 0.1, 0.1
            centroid_marker.color.a = 1.0
            if skeleton_i == msg.tracked_person_id:
                centroid_marker.color.r, centroid_marker.color.g, centroid_marker.color.b = (0.0,1.0,0.0)
            else:
                centroid_marker.color.r, centroid_marker.color.g, centroid_marker.color.b = (1.0,0.0,0.0)
            centroid_marker.pose.orientation.w = 1.0
            centroid_marker.pose.position.x = skel_centroid[0]
            centroid_marker.pose.position.y = skel_centroid[1]
            centroid_marker.pose.position.z = skel_centroid[2]
            centroid_marker.id = skeleton_i
            centroid_marker.lifetime = rospy.Duration(float(args.lifetime))
            centroid_marker.header.stamp = now
            skel_pub.publish(centroid_marker)

rospy.init_node('pose_visualizer')
skel_pub = rospy.Publisher('openpifpaf_markers', Marker, queue_size=100)
pose_sub = rospy.Subscriber('openpifpaf_pose_transformed', PoseEstimation, pose_cb)

rospy.spin()
