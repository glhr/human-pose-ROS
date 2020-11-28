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
from sensor_msgs.msg import Image, CameraInfo
from pose_utils.utils import *

from sympy.solvers import solve
from sympy import Symbol


logger=get_logger()
pp = get_printer()

cam = True
FRAME_ID = "/world"

rospy.init_node('pose_limbs')

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
parser.add_argument('--cam',
                 default='wrist')

args, unknown = parser.parse_known_args()

if args.cam in ["wrist","base"]:
    DEPTH_INFO_TOPIC = f'/{args.cam}_camera/camera/aligned_depth_to_color/camera_info'
    print(DEPTH_INFO_TOPIC)

connected_points = {
'eyeshoulder_left': (5,1),
'eyeshoulder_right': (6,2),
'earshoulder_left': (5,3),
'earshoulder_right': (6,4),
'eyear_left': (1,3),
'eyear_right': (2,4),
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
    'upperarm': 0.35,
    'calf': 0.55,
    'thigh': 0.6,
    'noseye': 0.1,
    'eyeshoulder': 0.4,
    'earshoulder': 0.25,
    'torso': 0.5,
    'eyear': 0.15
}

apply_constraints = ['upperarm','forearm','thigh','calf','noseye','eyear','earshoulder']

pairs = dict(list(enumerate(openpifpaf.datasets.constants.COCO_KEYPOINTS)))
pp.pprint(pairs)

colors = dict()
for k in range(100):
  colors[k] = tuple(np.random.randint(256, size=3)/256)


if args.cam in ["wrist","base"]:
    cameraInfo = rospy.wait_for_message(DEPTH_INFO_TOPIC, CameraInfo, timeout=2)
    logger.info("Got camera info")

def reposition_joint(ref_joint, old_joint, desired_length):
    x_ref, y_ref, z_ref = ref_joint[:3]
    x_old, y_old, z_old = old_joint[:3]

    print("Old joint {}".format([x_old, y_old, z_old.real], [x_ref, y_ref, z_ref.real]))

    a = 1
    b = -2*z_ref
    c = (x_old - x_ref)**2 + (y_old - y_ref)**2 + z_ref**2 - desired_length**2

    z_new_1 = (-b + np.sqrt(b**2 + 4*a*c+0j))/2*a
    z_new_2 = (-b + np.sqrt(b**2 - 4*a*c+0j))/2*a

    # x = Symbol('x')
    # z_new = float(min(solve((x_old-x_ref)**2 + (y_old-y_ref)**2 + (x-z_ref)**2 - desired_length**2, x)))
    z_new = min(z_new_1, z_new_2)

    new_joint = [x_old, y_old, z_new]

    print("--> New joint: {}".format(new_joint) )
    return new_joint

def pose_cb(msg):

    pose = msg
    person_id = msg.tracked_person_id

    for n, skeleton in enumerate(msg.skeletons):
        skeleton_dict = message_converter.convert_ros_message_to_dictionary(skeleton)

        for k, v in skeleton_dict.items():
            if isinstance(v,list) and len(v):
                if args.cam in ["wrist","base"]:
                    v[:3] = pixel_to_camera(cameraInfo, (v[0],v[1]), v[2])

        limbs_before, limbs_after = dict(), dict()
        for limb,conn in connected_points.items():
            label_1 = list(skeleton_dict.keys())[conn[0]]
            label_2 = list(skeleton_dict.keys())[conn[1]]
            if args.debug: logger.debug(f"{label_1} --> {label_2}")
            pnt_1 = skeleton_dict[label_1][:3]
            pnt_2 = skeleton_dict[label_2][:3]

            if len(pnt_1) and len(pnt_2):

                label = limb.split('_')[0]

                skeleton_dict[list(skeleton_dict.keys())[conn[1]]][:3] = pnt_2

                if label in apply_constraints:
                    distance = distance_between_points(pnt_1, pnt_2)
                    limbs_before[limb] = distance
                    thresh = thresholds[label]
                    if distance > thresh:
                        if args.cam in ["wrist","base"]:
                            pnt_2 = reposition_joint(ref_joint=pnt_1, old_joint=pnt_2, desired_length=thresh)
                        skeleton_dict[list(skeleton_dict.keys())[conn[1]]][:3] = pnt_2
                    limbs_after[limb] = distance_between_points(pnt_1, pnt_2)
                    # print(distance)

            pose.skeletons[n] = message_converter.convert_dictionary_to_ros_message("human_pose_ROS/Skeleton", skeleton_dict)

        if n == person_id:
            limb_before_pub.publish(message_converter.convert_dictionary_to_ros_message("human_pose_ROS/Limbs",limbs_before))
            limb_after_pub.publish(message_converter.convert_dictionary_to_ros_message("human_pose_ROS/Limbs",limbs_after))
            # print(limbs_before)

    skel_pub.publish(pose)





logger.warning("Using /openpifpaf_pose")
pose_sub = rospy.Subscriber('openpifpaf_pose', PoseEstimation, pose_cb)
limb_before_pub = rospy.Publisher('limbs_before', Limbs, queue_size=100)
limb_after_pub = rospy.Publisher('limbs_after', Limbs, queue_size=100)
skel_pub = rospy.Publisher('openpifpaf_pose_constrained_limbs', PoseEstimation, queue_size=100)

rospy.spin()
