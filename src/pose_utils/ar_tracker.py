#! /usr/bin/env python

import rospy
import numpy as np
import tf
from tf.transformations import quaternion_from_euler, quaternion_multiply
from human_pose_ROS.msg import Skeleton, PoseEstimation
from rospy_message_converter import message_converter
from std_msgs.msg import Float32
from vision_utils.logger import get_logger, get_printer
from vision_utils.timing import CodeTimer
from pose_utils.utils import get_points_centroid, angle_from_centroid, cam_to_world, distance_between_points, vector_from_2_points
from visualization_msgs.msg import Marker, MarkerArray
from human_pose_ROS.msg import Skeleton, PoseEstimation
import vg
import argparse

from pose_utils.utils import get_points_centroid, angle_from_centroid, cam_to_world, distance_between_points, vector_from_2_points

logger = get_logger()
pp = get_printer()

CAM_FRAME = "/wrist_camera_depth_optical_frame"

ar_point = []
pred_point = []
pred_updated = False

def skel_cb(msg):
    global ar_point, pred_point
    if msg.skeletons[0].dummy:
        ar_point = msg.skeletons[0].centroid
        logger.debug("AR point: {}".format(ar_point))
    else:
        if len(msg.skeletons[0].right_wrist):
            pred_point = msg.skeletons[0].right_wrist
            logger.debug("Pred point: {}".format(pred_point))

def ar_cb(msg):
    # tf_listener.waitForTransform('/world', CAM_FRAME, rospy.Time(), rospy.Duration(0.5))
    # (trans, rot) = tf_listener.lookupTransform('/world', CAM_FRAME, rospy.Time())
    # world_to_cam = tf.transformations.compose_matrix(translate=trans, angles=tf.transformations.euler_from_quaternion(rot))

    marker_pos = msg.pose.position
    # marker_tf = cam_to_world([marker_pos.x, marker_pos.y, marker_pos.z], world_to_cam)
    skel = Skeleton()
    pose = PoseEstimation()
    skel.centroid = [marker_pos.x, marker_pos.y, marker_pos.z]
    pose.skeletons.append(skel)
    pose_pub.publish(pose)
    # print(skel)


rospy.init_node("ar_test")

tf_listener = tf.TransformListener()
rospy.Subscriber("visualization_marker",Marker,ar_cb)
rospy.Subscriber("openpifpaf_pose_transformed",PoseEstimation,skel_cb)
pose_pub = rospy.Publisher("ar_skeleton",PoseEstimation, queue_size=1)
ar_distance_pub = rospy.Publisher("ar_distance",Float32, queue_size=1)

while not rospy.is_shutdown():
    if len(ar_point) and len(pred_point):
        distance = distance_between_points(ar_point, pred_point)
        ar_distance_pub.publish(distance)
        rospy.sleep(0.05)
