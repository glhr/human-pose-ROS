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

def points_cb(msg):
    with CodeTimer() as timer:
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
            for i,v in msg_dict.items():
                msg_dict_tf[i] = cam_to_world(v, world_to_cam)

            msg_tf = message_converter.convert_dictionary_to_ros_message("human_pose_ROS/Skeleton",msg_dict_tf)
            pose_tf.skeletons.append(msg_tf)
            # print(list(msg_dict_tf.values()))
            valid_points = [v for v in msg_dict_tf.values() if len(v)]

            centroids[skeleton_i] = get_points_centroid(list(valid_points))
            if centroids[skeleton_i] is not None:
                logger.debug("{} - Centroid: {}".format(skeleton_i, centroids[skeleton_i] ))
                distances[skeleton_i] = distance_between_points([0,0,0],centroids[skeleton_i])
                msg_tf.centroid = centroids[skeleton_i]
            msg_tf.id = skeleton.id


        logger.info("{} person(s) found".format(len(msg.skeletons)))
        if len(msg.skeletons):
            closest_skeleton_i = min(distances, key=distances.get)
            pose_tf.tracked_person_id = closest_skeleton_i
            pan_angle = angle_from_centroid(centroids[closest_skeleton_i], ref_vector=[0,1,0], normal_vector=[0,0,-1])

            centroid_v = vector_from_2_points(camera_point,centroids[closest_skeleton_i])
            ref_v = vector_from_2_points(camera_point,np.add(camera_point,[0,1,0]))
            tilt_angle = angle_from_centroid(centroid_v, ref_vector=ref_v, normal_vector=[1,0,0])
            logger.debug("--> Angle of closest person {}: pan {} tilt {}".format(closest_skeleton_i, pan_angle, tilt_angle))
            pan_pub.publish(pan_angle)
            tilt_pub.publish(tilt_angle)
            pose_pub.publish(pose_tf)
        else:
            pan_pub.publish(0.0)
            tilt_pub.publish(0.0)
    logger.info("Callback took {}ms".format(timer.took))


rospy.init_node("point_transform")

parser = argparse.ArgumentParser(description='arg for which human pose estimation to use (realsense or open)')
parser.add_argument('--realsense', dest='realsense', action='store_true')
args, unknown = parser.parse_known_args()

if args.realsense:
    pose_sub = rospy.Subscriber('realsense_pose', PoseEstimation, points_cb)
else:
    pose_sub = rospy.Subscriber('openpifpaf_pose', PoseEstimation, points_cb)

pose_pub = rospy.Publisher('openpifpaf_pose_transformed', PoseEstimation, queue_size=1)
pan_pub = rospy.Publisher('ref_pan_angle', Float32, queue_size=1)
tilt_pub = rospy.Publisher('ref_tilt_angle', Float32, queue_size=1)

tf_listener = tf.TransformListener()

rospy.spin()
