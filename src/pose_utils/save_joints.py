#! /usr/bin/env python3

import rospy
import numpy as np
from human_pose_ROS.msg import Skeleton, PoseEstimation
from rospy_message_converter import message_converter
from std_msgs.msg import Float32, String
from sensor_msgs.msg import Image, CameraInfo
from vision_utils.logger import get_logger, get_printer
from vision_utils.timing import CodeTimer
from vision_utils.img import image_to_numpy
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


pnts_dict = []

def save_everything():
    csv_columns = ['x','y','z']
    with open(Path.joinpath(project_path,f'joint_data/{time.time()}-test.csv'), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in pnts_dict:
            writer.writerow(data)


def points_cb(msg):
    global num_frames, num_images, images, frames

    if not saving:
        num_frames += 1
        if len(msg.skeletons):
            skeleton = msg.skeletons[0]
            msg_dict = message_converter.convert_ros_message_to_dictionary(skeleton)
            msg_dict = {k: v for k, v in msg_dict.items() if isinstance(v, list)}
            for i,v in msg_dict.items():
                if i == "left_shoulder":
                    # pnt1_cam = pixel_to_camera(cameraInfo, (v[0],v[1]), v[2])
                    pnt1_cam = v
                    pnts_dict.append({
                        'x': pnt1_cam[0],
                        'y': pnt1_cam[1],
                        'z': pnt1_cam[2]
                    })

            logger.info(num_frames)

pose_sub = rospy.Subscriber('openpifpaf_pose_transformed_pose_cam', PoseEstimation, points_cb)
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

rospy.spin()
