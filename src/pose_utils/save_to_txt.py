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
num_images = 0

frames = []
images = []

FRAMES_PER_SEQ = 30

if args.cam in ["wrist","base"]:
    logger.info("Waiting for camera info :)")
    cameraInfo = rospy.wait_for_message(DEPTH_INFO_TOPIC, CameraInfo)
    logger.info("Got camera info")

def save_frames():
    with open(Path.joinpath(project_path,f"txt/{num_frames}.txt"), "w") as text_file:
        lines = list(','.join(pnts) for pnts in frames)
        lines = '\n'.join(lines)
        print(f"{lines}", file=text_file)

def save_images():
    # with imageio.get_writer(Path.joinpath(project_path,f"txt/{num_frames}.gif"), mode='I') as writer:
    #     for path in images:
    #         image = imageio.imread(path)
    #         writer.append_data(image)

    with imageio.get_writer(Path.joinpath(project_path,f"txt/{num_images}.gif"), mode='I') as writer:
        for image in images:
            writer.append_data(image)


def img_cb(msg):
    global images, num_images
    # path = msg.data
    num_images += 1
    images.append(image_to_numpy(msg))


def points_cb(msg):
    global num_frames, num_images, images, frames

    num_frames += 1
    if len(msg.skeletons):
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

        logger.info(num_frames)

pose_sub = rospy.Subscriber('openpifpaf_pose_filtered', PoseEstimation, points_cb)
# img_sub = rospy.Subscriber('openpifpaf_savepath', String, img_cb)
poseimg_sub = rospy.Subscriber('openpifpaf_img', Image, img_cb)


while not rospy.is_shutdown():
    if len(frames)>FRAMES_PER_SEQ:
        frames = frames[-FRAMES_PER_SEQ:]
        save_frames()
        frames = []
        images = images[-FRAMES_PER_SEQ:]
        save_images()
        images = []
        logger.debug("Saving")
