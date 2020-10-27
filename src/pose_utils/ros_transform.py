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
import vg
import argparse

logger = get_logger()
pp = get_printer()

CAM_FRAME = "/wrist_camera_depth_optical_frame"
camera_point = [-0.0334744,-0.20912,1.85799]
ref_v = [0,1,0]


def points_cb(msg):
    with CodeTimer() as timer:
        if not args.norobot or not args.ar:
            tf_listener.waitForTransform('/world', CAM_FRAME, rospy.Time(), rospy.Duration(0.5))
            (trans, rot) = tf_listener.lookupTransform('/world', CAM_FRAME, rospy.Time())
            world_to_cam = tf.transformations.compose_matrix(translate=trans, angles=tf.transformations.euler_from_quaternion(rot))

        pose_tf = PoseEstimation()
        distances = dict()
        centroids = dict()

        for skeleton_i, skeleton in enumerate(msg.skeletons):
            msg_dict = message_converter.convert_ros_message_to_dictionary(skeleton)
            msg_dict.pop("id",None)
            msg_dict = {k: v for k, v in msg_dict.items() if len(v) and v[-1] > 0}
            msg_dict_tf = dict()
            if not args.norobot or not args.ar:
                for i,v in msg_dict.items():
                    msg_dict_tf[i] = cam_to_world(v, world_to_cam)
            else:
                msg_dict_tf = msg_dict

            msg_tf = message_converter.convert_dictionary_to_ros_message("human_pose_ROS/Skeleton",msg_dict_tf)
            print(msg_tf.centroid)
            centroids[skeleton_i] = msg_tf.centroid

            if len(centroids[skeleton_i]):
                distances[skeleton_i] = distance_between_points([0,0,0],centroids[skeleton_i])

            msg_tf.id = skeleton.id
            pose_tf.skeletons.append(msg_tf)

        if len(msg.skeletons):
            try:
                closest_skeleton_i = min(distances, key=distances.get)
                pose_tf.tracked_person_id = closest_skeleton_i
                pan_angle = angle_from_centroid(centroids[closest_skeleton_i], ref_vector=[0,1,0], normal_vector=[0,0,-1])

                centroid_v = vector_from_2_points(camera_point,centroids[closest_skeleton_i])
                tilt_angle = angle_from_centroid(centroid_v, ref_vector=ref_v, normal_vector=[1,0,0])
                if args.debug: logger.debug("--> Angle of closest person {}: pan {} tilt {}".format(closest_skeleton_i, pan_angle, tilt_angle))
                pan_pub.publish(pan_angle)
                tilt_pub.publish(tilt_angle)
            except Exception as e:
                logger.warning(e)
            pose_pub.publish(pose_tf)
        else:
            pan_pub.publish(200)
            tilt_pub.publish(200)
    logger.info("{} person(s) found, callback took {}ms".format(len(msg.skeletons), timer.took))


rospy.init_node("point_transform")

parser = argparse.ArgumentParser(description='arg for which human pose estimation to use (realsense or open)')
parser.add_argument('--realsense', dest='realsense', action='store_true')
parser.add_argument('--norobot', dest='norobot', action='store_true')
parser.add_argument('--debug',
                 action='store_true',
                 help='Print transform debug')
parser.add_argument('--ar',
                 action='store_true',
                 help='Use AR dummy marker')
args, unknown = parser.parse_known_args()

if args.realsense:
    pose_sub = rospy.Subscriber('realsense_pose', PoseEstimation, points_cb)
elif args.ar:
    pose_sub = rospy.Subscriber('ar_skeleton', PoseEstimation, points_cb)
else:
    pose_sub = rospy.Subscriber('openpifpaf_pose_filtered', PoseEstimation, points_cb)

pose_pub = rospy.Publisher('openpifpaf_pose_transformed', PoseEstimation, queue_size=1)
pan_pub = rospy.Publisher('ref_pan_angle', Float32, queue_size=1)
tilt_pub = rospy.Publisher('ref_tilt_angle', Float32, queue_size=1)

tf_listener = tf.TransformListener()

rospy.spin()
