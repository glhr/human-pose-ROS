#! /usr/bin/env python3

import rospy
import numpy as np
from rospy_message_converter import message_converter
from std_msgs.msg import Float32
from vision_utils.logger import get_logger, get_printer
from vision_utils.timing import CodeTimer
from pose_utils.utils import get_points_centroid, angle_from_centroid, cam_to_world, distance_between_points, vector_from_2_points
from visualization_msgs.msg import Marker, MarkerArray
from human_pose_ROS.msg import Skeleton, PoseEstimation
from tracking.kalman_utils import uncertainty_gain
import vg
import argparse

from pose_utils.utils import *

logger = get_logger()
pp = get_printer()

parser = argparse.ArgumentParser()
parser.add_argument('--debug',
                    default=False,
                 action='store_true',
                 help='Print transform debug')
args, unknown = parser.parse_known_args()

camera_point = [-0.0334744,-0.20912,1.85799]
ref_v = [0,1,0]
history = dict()

def skel_cb(msg):
    global history

    centroids = dict()
    distances = dict()

    msg_dict = message_converter.convert_ros_message_to_dictionary(msg.skeletons[0])

    centroid = msg_dict["centroid"][0:3]
    uncertainty = msg_dict["centroid"][3:6]

    uncertainty_value = max(uncertainty)

    if uncertainty_value < 1:

        pan_angle = angle_from_centroid(centroid, ref_vector=[0,1,0], normal_vector=[0,0,-1])

        centroid_v = vector_from_2_points(camera_point,centroid)
        tilt_angle = angle_from_centroid(centroid_v, ref_vector=ref_v, normal_vector=[1,0,0])
        if args.debug: logger.debug("--> Angle of closest person {}: pan {} tilt {}".format(closest_skeleton_i, pan_angle, tilt_angle))
        pan_pub.publish(pan_angle)
        tilt_pub.publish(tilt_angle)
    else:
        logger.info(f"Uncertainty is >= {uncertainty_value}")
        pan_pub.publish(200)
        tilt_pub.publish(200)

rospy.init_node("get_centroid")

rospy.Subscriber("openpifpaf_pose_transformed_kalman",PoseEstimation,skel_cb)
pan_pub = rospy.Publisher('ref_pan_angle', Float32, queue_size=1)
tilt_pub = rospy.Publisher('ref_tilt_angle', Float32, queue_size=1)

rospy.spin()
