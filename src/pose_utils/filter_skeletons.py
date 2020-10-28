#!/usr/bin/env python3
import rospy
from std_msgs.msg import Int32
from human_pose_ROS.msg import Skeleton, PoseEstimation
from rospy_message_converter import message_converter
import numpy as np
from vision_utils.logger import get_logger, get_printer
logger = get_logger()
from pose_utils.utils import *
import argparse

history = dict()
window_length = 50

filter = np.mean

parser = argparse.ArgumentParser(description='Directory of PNG images to use for inference.')
parser.add_argument('--nofilter', action='store_true')
parser.add_argument('--debug',
                    default=False,
                 action='store_true',
                 help='Print transform debug')

args, unknown = parser.parse_known_args()

def skel_callback(msg):
    global history
    pose_filtered = PoseEstimation()
    pose_filtered.tracked_person_id = msg.tracked_person_id
    centroids = dict()
    if args.debug: logger.info(f"Filter: found {len(msg.skeletons)} skeletons")

    for skeleton_i, skeleton in enumerate(msg.skeletons):
        skel_filtered = dict()
        msg_dict = message_converter.convert_ros_message_to_dictionary(skeleton)
        if not history.get(skeleton_i,0):
            history[skeleton_i] = dict()
        msg_dict.pop("id",None)
        msg_dict.pop("dummy",None)


        if args.nofilter or skeleton.dummy:
            pose_filtered.skeletons.append(skeleton)
            continue
        for joint,kp in msg_dict.items():
            if len(kp):
                if not history[skeleton_i].get(joint,0):
                    history[skeleton_i][joint] = [kp]
                else:
                    history[skeleton_i][joint].append(kp)
                if len(history[skeleton_i][joint]) > window_length:
                    history[skeleton_i][joint] = history[skeleton_i][joint][-window_length:]
                # average_x = filter([i[0] for i in history[skeleton_i][joint]][-3:])
                # average_y = filter([i[1] for i in history[skeleton_i][joint]][-3:])
                average_z = np.median([i[2] for i in history[skeleton_i][joint]][-10:])
                skel_filtered[joint] = (kp[0], kp[1], average_z)
        skel_filtered["id"] = skeleton_i
        pose_filtered.skeletons.append(message_converter.convert_dictionary_to_ros_message("human_pose_ROS/Skeleton",skel_filtered))
    pose_filtered_pub.publish(pose_filtered)
    pose_raw_pub.publish(msg)
    if args.debug: logger.info(f"Filter: outputting {len(pose_filtered.skeletons)} skeletons")

rospy.init_node('average_skeletons')
rospy.Subscriber('/openpifpaf_pose', PoseEstimation, skel_callback)
pose_filtered_pub = rospy.Publisher('/openpifpaf_pose_filtered', PoseEstimation, queue_size=1)
pose_raw_pub = rospy.Publisher('/openpifpaf_pose_raw', PoseEstimation, queue_size=1)
rospy.spin()
