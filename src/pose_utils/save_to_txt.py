#! /usr/bin/env python3

import rospy
import numpy as np
from human_pose_ROS.msg import Skeleton, PoseEstimation
from rospy_message_converter import message_converter
from std_msgs.msg import Float32
from sensor_msgs.msg import Image, CameraInfo
from vision_utils.logger import get_logger, get_printer
from vision_utils.timing import CodeTimer
from pose_utils.utils import *
import vg
import argparse
from pathlib import Path

project_path = Path(__file__).parent.absolute()

logger = get_logger()
pp = get_printer()

parser = argparse.ArgumentParser()
parser.add_argument('--debug',
                    default=True,
                 action='store_true',
                 help='Print transform debug')
parser.add_argument('--cam', default='wrist')
args, unknown = parser.parse_known_args()

rospy.init_node("save_pose")

save_points = ["eye", "nose", "shoulder", "elbow", "wrist"]

DEPTH_INFO_TOPIC = '/{}_camera/camera/aligned_depth_to_color/camera_info'.format(args.cam)

im_h = 480
im_w = 848

num_frames = 0

frames = []

if args.cam in ["wrist","base"]:
    logger.info("Waiting for camera info :)")
    cameraInfo = rospy.wait_for_message(DEPTH_INFO_TOPIC, CameraInfo)
    logger.info("Got camera info")


def save_frames():
    with open(Path.joinpath(project_path,f"txt/{num_frames}.txt"), "w") as text_file:
        lines = list(','.join(pnts) for pnts in frames)
        lines = '\n'.join(lines)
        print(f"{lines}", file=text_file)

def points_cb(msg):
    global num_frames, frames

    if len(frames)>10:
        frames = frames[-10:]
        save_frames()

    skeleton = msg.skeletons[0]
    msg_dict = message_converter.convert_ros_message_to_dictionary(skeleton)
    msg_dict = {k: v for k, v in msg_dict.items() if isinstance(v, list) and k.split("_")[-1] in save_points}
    msg_dict_tf = dict()
    pnts = []
    for i,v in msg_dict.items():
        if len(v) and args.cam in ["wrist","base"]:
            pnt1_cam = pixel_to_camera(cameraInfo, (v[0],v[1]), v[2])
        elif len(v):
            pnt1_cam = v
        else:
            pnt1_cam = [0,0,0]
        msg_dict_tf[i] = pnt1_cam
        pnts.extend(pnt1_cam)

    frames.append([str(i) for i in pnts])
    print(len(pnts))

    num_frames += 1
    logger.info(num_frames)


pose_sub = rospy.Subscriber('openpifpaf_pose_filtered', PoseEstimation, points_cb)


rospy.spin()
