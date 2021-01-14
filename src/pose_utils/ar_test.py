#! /usr/bin/env python3

import rospy
import numpy as np
from human_pose_ROS.msg import Skeleton, PoseEstimation
from rospy_message_converter import message_converter
from std_msgs.msg import Float32, String
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose
from vision_utils.logger import get_logger, get_printer
from vision_utils.timing import CodeTimer
from vision_utils.img import image_to_numpy
from visualization_msgs.msg import Marker, MarkerArray
from pose_utils.utils import *
import vg
import argparse
from pathlib import Path
import imageio
import time

project_path = Path(__file__).parent.absolute()

logger = get_logger()
pp = get_printer()

parser = argparse.ArgumentParser()
parser.add_argument('--debug',
                    default=True,
                 action='store_true')
parser.add_argument('--cam', default='wrist')
args, unknown = parser.parse_known_args()

rospy.init_node("save_joints")

DEPTH_INFO_TOPIC = '/{}_camera/camera/aligned_depth_to_color/camera_info'.format(args.cam)

num_frames = 0
saving = False


import os
import csv
if not os.path.exists(Path.joinpath(project_path,f"joint_data")):
    os.makedirs(Path.joinpath(project_path,f"joint_data"))

if args.cam in ["wrist","base"]:
    logger.info("Waiting for camera info :)")
    cameraInfo = rospy.wait_for_message(DEPTH_INFO_TOPIC, CameraInfo)
    logger.info("Got camera info")


pnts_dict = dict()

joint_of_interest = "left_wrist"
ar_position = []
skel_position = []

def save_everything():
    csv_columns = ['x','y','z']
    t = time.time()
    # for joint in pnts_dict:
    #     with open(Path.joinpath(project_path,f'joint_data/{t}-{joint}.csv'), 'w') as csvfile:
    #         writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    #         writer.writeheader()
    #         for data in pnts_dict[joint]:
    #             writer.writerow(data)


def points_cb(msg):
    global num_frames, frames, skel_position

    num_frames += 1
    if len(msg.skeletons):
        skeleton = msg.skeletons[0]
        msg_dict = message_converter.convert_ros_message_to_dictionary(skeleton)
        msg_dict = {k: v for k, v in msg_dict.items() if isinstance(v, list)}
        for i,v in msg_dict.items():
            # pnt1_cam = pixel_to_camera(cameraInfo, (v[0],v[1]), v[2])
            if i == joint_of_interest and len(v):
                pnt1_cam = v[:3]
                pnts_dict[i] = pnts_dict.get(i,[])
                pnts_dict[i].append({
                    'x': pnt1_cam[0],
                    'y': pnt1_cam[1],
                    'z': pnt1_cam[2]
                })
                skel_position = pnt1_cam

        logger.info(num_frames)

def ar_cb(msg):
    global ar_position
    # tf_listener.waitForTransform('/world', CAM_FRAME, rospy.Time(), rospy.Duration(0.5))
    # (trans, rot) = tf_listener.lookupTransform('/world', CAM_FRAME, rospy.Time())
    # world_to_cam = tf.transformations.compose_matrix(translate=trans, angles=tf.transformations.euler_from_quaternion(rot))

    marker_pos = msg.pose.position
    ar_position = [marker_pos.x, marker_pos.y, marker_pos.z]
    logger.info("marker :D")

    # print(skel)

pose_sub = rospy.Subscriber('openpifpaf_pose_transformed_pose_cam', PoseEstimation, points_cb)
rospy.Subscriber("visualization_marker",Marker,ar_cb)
ar_pub = rospy.Publisher("ar_skeleton",Pose, queue_size=1)
# img_sub = rospy.Subscriber('openpifpaf_savepath', String, img_cb)

import signal, sys


def signal_handler(sig, frame):
    global saving
    if not saving:
        saving = True
        logger.warning("Saving everything")
        save_everything()
        sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

while not rospy.is_shutdown():
    if len(ar_position) and len(skel_position):

        pose = Pose()
        pose.position.x = ar_position[0] - skel_position[0]
        pose.position.y = ar_position[1] - skel_position[1]
        pose.position.z = ar_position[2] - skel_position[2]
        ar_pub.publish(pose)
    rospy.sleep(0.1)
        # print(f"x:{pose.position.x} y:{pose.position.y} z:{pose.position.z}")
