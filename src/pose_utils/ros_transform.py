#! /usr/bin/env python

import rospy
import numpy as np
import tf
from tf.transformations import quaternion_from_euler, quaternion_multiply
from human_pose_ROS.msg import Skeleton, PoseEstimation
from rospy_message_converter import message_converter
from std_msgs.msg import Float32
from sensor_msgs.msg import Image, CameraInfo
from vision_utils.logger import get_logger, get_printer
from vision_utils.timing import CodeTimer
from pose_utils.utils import *
import vg
import argparse

logger = get_logger()
pp = get_printer()

parser = argparse.ArgumentParser()
parser.add_argument('--realsense', dest='realsense', action='store_true')
parser.add_argument('--norobot', dest='norobot', action='store_true')
parser.add_argument('--cam', default='wrist')
parser.add_argument('--debug',
                    default=False,
                 action='store_true',
                 help='Print transform debug')
parser.add_argument('--ar',
                 action='store_true',
                 help='Use AR dummy marker')
args, unknown = parser.parse_known_args()

CAM_FRAME = "/{}_camera_depth_optical_frame".format(args.cam)
DEPTH_INFO_TOPIC = '/{}_camera/camera/aligned_depth_to_color/camera_info'.format(args.cam)

im_h = 480
im_w = 848

rospy.init_node("point_transform")

if args.cam in ["wrist","base"]:
    cameraInfo = rospy.wait_for_message(DEPTH_INFO_TOPIC, CameraInfo, timeout=3)
    logger.info("Got camera info")

def ar_cb(msg):
    with CodeTimer() as timer:
        pose_tf = PoseEstimation()

        if args.cam in ["wrist","base"] and (not args.norobot or not args.ar):
            tf_listener.waitForTransform('/world', CAM_FRAME, rospy.Time(), rospy.Duration(0.5))
            (trans, rot) = tf_listener.lookupTransform('/world', CAM_FRAME, rospy.Time())
            world_to_cam = tf.transformations.compose_matrix(translate=trans, angles=tf.transformations.euler_from_quaternion(rot))

        for skeleton_i, skeleton in enumerate(msg.skeletons):
            msg_dict = message_converter.convert_ros_message_to_dictionary(skeleton)
            msg_dict.pop("id",None)
            msg_dict.pop("dummy",None)
            msg_dict = {k: v for k, v in msg_dict.items() if len(v) and v[-1] > 0}
            msg_dict_tf = dict()
            if args.cam in ["wrist","base"]:
                for i,v in msg_dict.items():
                    pnt1_cam = pixel_to_camera(cameraInfo, (v[0],v[1]), v[2])
                    msg_dict_tf[i] = cam_to_world(pnt1_cam, world_to_cam)
            else:
                msg_dict_tf = msg_dict

            msg_tf = message_converter.convert_dictionary_to_ros_message("human_pose_ROS/Skeleton",msg_dict_tf)
            if args.debug: print(msg_tf.centroid)

            msg_tf.id = -1
            msg_tf.dummy = True
            pose_tf.skeletons.append(msg_tf)

            pose_pub.publish(pose_tf)
    logger.info("{} AR Tracker skeleton found, callback took {}ms".format(len(msg.skeletons), timer.took))



def points_cb(msg):
    with CodeTimer() as timer:
        if args.cam in ["wrist","base"] and (not args.norobot or not args.ar):
            tf_listener.waitForTransform('/world', CAM_FRAME, rospy.Time(), rospy.Duration(0.5))
            (trans, rot) = tf_listener.lookupTransform('/world', CAM_FRAME, rospy.Time())
            world_to_cam = tf.transformations.compose_matrix(translate=trans, angles=tf.transformations.euler_from_quaternion(rot))

        pose_tf = PoseEstimation()


        for skeleton_i, skeleton in enumerate(msg.skeletons):
            msg_dict = message_converter.convert_ros_message_to_dictionary(skeleton)
            msg_dict.pop("id",None)
            msg_dict.pop("dummy",None)
            msg_dict = {k: v for k, v in msg_dict.items() if len(v) and v[-1] > 0}
            msg_dict_tf = dict()
            if args.cam in ["wrist","base"] and (not args.norobot or not args.ar):
                for i,v in msg_dict.items():
                    # pnt1_cam = pixel_to_camera(cameraInfo, (v[0],v[1]), v[2])
                    msg_dict_tf[i] = cam_to_world(v[0:3], world_to_cam)
            else:
                msg_dict_tf = msg_dict

            for i,v in msg_dict.items():
                msg_dict_tf[i].extend(v[3:6])
                print(msg_dict_tf[i])

            msg_tf = message_converter.convert_dictionary_to_ros_message("human_pose_ROS/Skeleton",msg_dict_tf)

            msg_tf.id = skeleton.id
            pose_tf.skeletons.append(msg_tf)

        pose_pub.publish(pose_tf)
    if args.debug: logger.info("{} person(s) found, callback took {}ms".format(len(msg.skeletons), timer.took))


if args.realsense:
    pose_sub = rospy.Subscriber('realsense_pose', PoseEstimation, points_cb)
ar_sub = rospy.Subscriber('ar_skeleton', PoseEstimation, ar_cb)
pose_sub = rospy.Subscriber('openpifpaf_pose_kalman', PoseEstimation, points_cb)

pose_pub = rospy.Publisher('openpifpaf_pose_transformed', PoseEstimation, queue_size=1)

if args.cam in ["wrist","base"]:
    tf_listener = tf.TransformListener()

rospy.spin()
