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
import cv2
import time

project_path = Path(__file__).parent.absolute()

logger = get_logger()
pp = get_printer()

parser = argparse.ArgumentParser()
parser.add_argument('--debug',
                    default=True,
                 action='store_true')
parser.add_argument('--person',
                    default='gala')
parser.add_argument('--action',
                    default='waving')
parser.add_argument('--example',
                    default=1)
parser.add_argument('--seqlength',
                    default=20)
parser.add_argument('--cam', default='wrist')
args, unknown = parser.parse_known_args()

rospy.init_node("save_pose")

save_points = ["eye", "nose", "shoulder", "elbow", "wrist"]

DEPTH_INFO_TOPIC = '/{}_camera/camera/aligned_depth_to_color/camera_info'.format(args.cam)

im_h = 480
im_w = 848

num_frames = 0
num_images = 0

saving = False

joint_coords = []
images = []

found_person = 0

FRAMES_PER_SEQ = args.seqlength

import os
if not os.path.exists(Path.joinpath(project_path,f"action_data")):
    os.makedirs(Path.joinpath(project_path,f"action_data"))

def save_everything():
    save_images(images)

    logger.warning(f"Person found in {found_person} out of {num_frames} frames")

def save_frames(pnts_list, n):
    with open(Path.joinpath(project_path,f"action_data/{args.person}-{args.action}-{args.example}-{n}.txt"), "w") as text_file:
        lines = list(','.join(pnts) for pnts in pnts_list)
        lines = '\n'.join(lines)
        print(f"{lines}", file=text_file)

def save_images(images):
    # with imageio.get_writer(Path.joinpath(project_path,f"txt/{num_frames}.gif"), mode='I') as writer:
    #     for path in images:
    #         image = imageio.imread(path)
    #         writer.append_data(image)

    # with imageio.get_writer(Path.joinpath(project_path,f"test.gif"), mode='I') as writer:
    #     for i,coords in enumerate(joint_coords):
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         image = images[i]
    #         try:
    #             if len(coords):
    #                 pt1 = tuple(coords[:2])
    #                 pt2 = tuple(coords[-2:])
    #                 print(pt1, pt2)
    #                 image = cv2.rectangle(image, pt1, pt2, (255,0,0), 2)
    #         except IndexError as e:
    #             print(e)
    #         writer.append_data(image)

    height,width,layers=images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'H264')

    video=cv2.VideoWriter(f"persondetection-{time.time()}.mp4",fourcc,10,(width,height))

    for i,coords in enumerate(joint_coords):
        font = cv2.FONT_HERSHEY_SIMPLEX
        try:
            image = images[i]
            if len(coords):
                pt1 = tuple(coords[:2])
                pt2 = tuple(coords[-2:])
                print(pt1, pt2)
                image = cv2.rectangle(image, pt1, pt2, (0,255,0), 2)
            video.write(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        except IndexError as e:
            print(e)

    video.release()


def img_cb(msg):
    global images, num_images
    if not saving:
        # path = msg.data
        num_images += 1
        images.append(image_to_numpy(msg))



def points_cb(msg):
    global num_frames, num_images, images, joint_coords, found_person

    if not saving:
        num_frames += 1
        if len(msg.skeletons):
            skeleton = msg.skeletons[0]
            msg_dict = message_converter.convert_ros_message_to_dictionary(skeleton)
            msg_dict = {k: v for k, v in msg_dict.items() if isinstance(v, list) and k.split("_")[-1] in save_points}
            msg_dict_tf = dict()
            pnts = []
            x = []
            y = []
            for i,v in msg_dict.items():
                if len(v):
                    x.append(int(v[0]))
                    y.append(int(v[1]))

            if len(x) and len(y):
                joint_coords.append([min(x), min(y), max(x), max(y)])
                found_person += 1
            else:
                joint_coords.append([])
            print(joint_coords)

        else:
            joint_coords.append([])
        logger.info(num_frames)

pose_sub = rospy.Subscriber('openpifpaf_pose', PoseEstimation, points_cb)
# img_sub = rospy.Subscriber('openpifpaf_savepath', String, img_cb)
poseimg_sub = rospy.Subscriber('openpifpaf_img', Image, img_cb)

import signal, sys


def signal_handler(sig, frame):
    global saving
    if not saving:
        saving = True
        logger.warning("Saving everything")
        save_everything()
        sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

sequence_number = 0

rospy.spin()
