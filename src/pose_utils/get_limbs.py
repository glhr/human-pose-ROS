#! /usr/bin/env python3
import openpifpaf
import rospy
import numpy as np
from human_pose_ROS.msg import Skeleton, PoseEstimation, Limbs
from rospy_message_converter import message_converter

from vision_utils.img import image_to_numpy, numpy_to_image, load_image
from vision_utils.logger import get_logger, get_printer
from vision_utils.timing import get_timestamp

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from pose_utils.utils import *

logger=get_logger()
pp = get_printer()

cam = True
FRAME_ID = "/world"

connected_points = {
'eyeshoulder_left': (1,5),
'eyeshoulder_right': (2,6),
'noseye_left': (0,1),
'noseye_right': (0,2),
'torso_left': (5,11),
'torso_right': (6,12),
'torso_top': (5,6),
'torso_bottom': (11,12),
'upperarm_left': (5,7),
'upperarm_right': (6,8),
'forearm_left': (7,9),
'forearm_right': (8,10),
'thigh_left':(11,13),
'thigh_right':(12,14),
'calf_left':(13,15),
'calf_right':(14,16)
}

thresholds = {
    'forearm': 0.35,
    'upperarm': 0.4,
    'calf': 0.55,
    'thigh': 0.55,
    'noseye': 0.1,
    'eyeshoulder': 0.4,
    'torso': 0.5
}

apply_constraints = ['forearm','upperarm','calf', 'noseye']

pairs = dict(list(enumerate(openpifpaf.datasets.constants.COCO_KEYPOINTS)))
pp.pprint(pairs)

colors = dict()
for k in range(100):
  colors[k] = tuple(np.random.randint(256, size=3)/256)

import argparse
parser = argparse.ArgumentParser(description='Visualization options')
parser.add_argument('--lifetime',
                 default=1,
                 help='Marker lifetime')
parser.add_argument('--filter',
                 action='store_true',
                 help='Use filtered poses')
parser.add_argument('--debug',
                 action='store_true',
                 help='Print visualization debug')

args, unknown = parser.parse_known_args()

def pose_cb(msg):

    pose = msg
    person_id = msg.tracked_person_id

    for n, skeleton in enumerate(msg.skeletons):
        skeleton_dict = message_converter.convert_ros_message_to_dictionary(skeleton)
        limbs_before, limbs_after = dict(), dict()
        for limb,conn in connected_points.items():
            label_1 = list(skeleton_dict.keys())[conn[0]]
            label_2 = list(skeleton_dict.keys())[conn[1]]
            if args.debug: logger.debug(f"{label_1} --> {label_2}")
            pnt_1 = skeleton_dict[label_1]
            pnt_2 = skeleton_dict[label_2]

            label = limb.split('_')[0]

            if len(pnt_1) and len(pnt_2) and label in apply_constraints:
                distance = distance_between_points(pnt_1, pnt_2)
                limbs_before[limb] = distance
                thresh = thresholds[label]
                if distance > thresh:
                    pnt_2 = pnt_1 + [thresh * p for p in pnt_2]/np.linalg.norm(pnt_2)
                    skeleton_dict[list(skeleton_dict.keys())[conn[1]]] = pnt_2
                limbs_after[limb] = distance_between_points(pnt_1, pnt_2)
                # print(distance)

            pose.skeletons[n] = message_converter.convert_dictionary_to_ros_message("human_pose_ROS/Skeleton", skeleton_dict)

        if n == person_id:
            limb_before_pub.publish(message_converter.convert_dictionary_to_ros_message("human_pose_ROS/Limbs",limbs_before))
            limb_after_pub.publish(message_converter.convert_dictionary_to_ros_message("human_pose_ROS/Limbs",limbs_after))
            # print(limbs_before)

    skel_pub.publish(pose)


rospy.init_node('pose_limbs')


logger.warning("Using /openpifpaf_pose_transformed")
pose_sub = rospy.Subscriber('openpifpaf_pose_transformed', PoseEstimation, pose_cb)
limb_before_pub = rospy.Publisher('limbs_before', Limbs, queue_size=100)
limb_after_pub = rospy.Publisher('limbs_after', Limbs, queue_size=100)
skel_pub = rospy.Publisher('openpifpaf_pose_constrained_limbs', PoseEstimation, queue_size=100)

rospy.spin()
