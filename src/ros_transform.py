#! /usr/bin/env python

import rospy
import numpy as np
import tf
from tf.transformations import quaternion_from_euler, quaternion_multiply
from human_pose_ROS.msg import Skeleton, PoseEstimation
from rospy_message_converter import message_converter
from std_msgs.msg import Float32
from vision_utils.logger import get_logger, get_printer
import vg

logger = get_logger()
pp = get_printer()

def points_cb(msg):

    pose_tf = PoseEstimation()
    for skeleton in msg.skeletons:
        msg_dict = message_converter.convert_ros_message_to_dictionary(skeleton)
        msg_dict_tf = dict()
        for i,v in msg_dict.items():
            if len(v):
                msg_dict_tf[i] = cam_to_world(v)
            else:
                msg_dict_tf[i] = []

        msg_tf = message_converter.convert_dictionary_to_ros_message("human_pose_ROS/Skeleton",msg_dict_tf)
        pose_tf.skeletons.append(msg_tf)
        print(list(msg_dict_tf.values()))
        valid_points = [v for v in msg_dict_tf.values() if len(v)]
        skel_centroid = get_points_centroid(list(valid_points))
        logger.debug("Centroid: {}".format(skel_centroid))
        angle = angle_from_centroid(skel_centroid)
        print(angle)
        angle_pub.publish(angle)

    pose_pub.publish(pose_tf)






def cam_to_world(cam_point, cam_frame="/wrist_camera_depth_optical_frame"):
    """Convert from camera_frame to world_frame

    Keyword arguments:
    cam_pose   -- PoseStamped from camera view
    cam_frame  -- The frame id of the camera
    """
    # cam_point = np.array([cam_pose[0], cam_pose[1], cam_pose[2]])
    tf_listener = tf.TransformListener()
    tf_listener.waitForTransform('/world', cam_frame, rospy.Time(), rospy.Duration(4))
    (trans, rot) = tf_listener.lookupTransform('/world', cam_frame, rospy.Time())

    world_to_cam = tf.transformations.compose_matrix(translate=trans, angles=tf.transformations.euler_from_quaternion(rot))
    obj_vector = np.concatenate((cam_point, np.ones(1))).reshape((4, 1))
    world_point = np.dot(world_to_cam, obj_vector)

    world_point = [p[0] for p in world_point]
    return world_point[0:3]

def get_points_centroid(arr):
    length = len(arr)
    arr = np.array(arr)
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    sum_z = np.sum(arr[:, 2])
    return sum_x/length, sum_y/length, sum_z/length

def angle_from_centroid(centroid):
    v0 = np.array([0,1,0])
    vcentroid = np.array(centroid)
    angle = vg.signed_angle(v0, vcentroid, look=np.array([0,0,-1]))
    logger.debug("Centroid angle: {}".format(angle))
    return angle


rospy.init_node("point_transform")

pose_sub = rospy.Subscriber('openpifpaf_pose', PoseEstimation, points_cb)
pose_pub = rospy.Publisher('openpifpaf_pose_transformed', PoseEstimation, queue_size=1)
angle_pub = rospy.Publisher('person_angle', Float32, queue_size=1)

rospy.spin()
