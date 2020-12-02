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
parser.add_argument('--debug', default=True, action='store_true')
parser.add_argument('--transform', action='store_true', default=False)
parser.add_argument('--cam', default='wrist')
args, unknown = parser.parse_known_args()

rospy.init_node("save_pose")

save_points = ["eye", "nose", "shoulder", "elbow", "wrist"]

DEPTH_INFO_TOPIC = '/{}_camera/camera/aligned_depth_to_color/camera_info'.format(args.cam)

num_frames = {
    'kalman': 0,
    'raw': 0
}
num_images = {
    'kalman': 0,
    'raw': 0
}

saving = False

joint_coords = {
    'kalman': [],
    'raw': []
}
images = []

found_person = {
    'kalman': 0,
    'raw': 0
}
new_image = {
    'kalman': False,
    'raw': False
}

MAX_UNCERTAINTY = None

logger.info("Waiting for camera info :)")
cameraInfo = rospy.wait_for_message(DEPTH_INFO_TOPIC, CameraInfo)
logger.info("Got camera info")

import os
if not os.path.exists(Path.joinpath(project_path,f"action_data")):
    os.makedirs(Path.joinpath(project_path,f"action_data"))

def save_everything():
    timestamp = time.time()
    save_images(images, name=timestamp)

    logger.warning(f"Person found in {found_person} out of {num_frames} frames")

def save_images(images, name=time.time()):
    height,width,layers=images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'H264')

    video=cv2.VideoWriter(f"persondetection-{name}.mp4",fourcc,10,(width,height))

    for i,(coords_kalman, coords_raw) in enumerate(zip(joint_coords['kalman'],joint_coords['raw'])):
        print(coords_raw, coords_kalman)
        font = cv2.FONT_HERSHEY_SIMPLEX
        try:
            image = images[i]
            if len(coords_raw):

                pt1 = list(coords_raw[:2])
                pt2 = list(coords_raw[-3:-1])
                color = (0,255,0)

                pt1 = tuple(int(i) for i in pt1)
                pt2 = tuple(int(i) for i in pt2)

                print(pt1, pt2)
                image_bb = cv2.rectangle(image.copy(), pt1, pt2, color, 2)
            else:
                image_bb = image
            if len(coords_kalman):
                pt1 = list(coords_kalman[:3])
                pt2 = list(coords_kalman[-3:])
                pt1 = camera_to_pixel(cameraInfo, pt1)
                pt2 = camera_to_pixel(cameraInfo, pt2)
                pt1 = tuple(int(i) for i in pt1)
                pt2 = tuple(int(i) for i in pt2)

                color = (0,0,255)
                print(pt1, pt2)
                image_bb = cv2.rectangle(image_bb.copy(), pt1, pt2, color, 2)

            video.write(cv2.cvtColor(image_bb, cv2.COLOR_BGR2RGB))
        except IndexError as e:
            print(e)

    video.release()

def img_cb(msg):
    global images, num_images, new_image
    if not saving:
        # path = msg.data
        num_images['kalman'] += 1
        num_images['raw'] += 1
        images.append(image_to_numpy(msg))
        new_image['kalman'] = True
        new_image['raw'] = True

def points_raw_cb(msg):
    global num_frames, num_images, images, joint_coords, found_person, new_image

    if not saving:
        new_image['raw'] = False
        num_frames['raw'] += 1
        if len(msg.skeletons):
            skeleton = msg.skeletons[0]
            msg_dict = message_converter.convert_ros_message_to_dictionary(skeleton)
            msg_dict = {k: v for k, v in msg_dict.items() if isinstance(v, list) and k.split("_")[-1] in save_points}
            x = []
            y = []
            z = []
            for i,v in msg_dict.items():
                if len(v):
                    uncertainty = v[3:]
                    x.append(v[0])
                    y.append(v[1])
                    z.append(v[2])

            if len(x) and len(y):
                joint_coords['raw'].append([min(x), min(y),  min(z), max(x), max(y), max(z)])
                found_person['raw'] += 1
            else:
                joint_coords['raw'].append([])
            print(joint_coords['raw'])

        else:
            joint_coords['raw'].append([])
        logger.info(num_frames['raw'])

def points_kalman_cb(msg):
    global num_frames, num_images, images, joint_coords, found_person, new_image

    if not saving and new_image['kalman']:
        new_image['kalman'] = False
        num_frames['kalman'] += 1
        if len(msg.skeletons):
            skeleton = msg.skeletons[0]
            msg_dict = message_converter.convert_ros_message_to_dictionary(skeleton)
            msg_dict = {k: v for k, v in msg_dict.items() if isinstance(v, list) and k.split("_")[-1] in save_points}
            x = []
            y = []
            z = []
            for i,v in msg_dict.items():
                if len(v):
                    uncertainty = v[3:]
                    if MAX_UNCERTAINTY is None or max(uncertainty) < MAX_UNCERTAINTY:
                        x.append(v[0])
                        y.append(v[1])
                        z.append(v[2])

            if len(x) and len(y):
                joint_coords['kalman'].append([min(x), min(y),  min(z), max(x), max(y), max(z)])
                found_person['kalman'] += 1
            else:
                joint_coords['kalman'].append([])
            print(joint_coords['kalman'])

        else:
            joint_coords['kalman'].append([])
        logger.info(num_frames['kalman'])

pose_kalman_sub = rospy.Subscriber('openpifpaf_pose_kalman', PoseEstimation, points_kalman_cb)
pose_raw_sub = rospy.Subscriber('openpifpaf_pose', PoseEstimation, points_raw_cb)
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
