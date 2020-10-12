#!/usr/bin/env python3
from cubemos.core.nativewrapper import CM_TargetComputeDevice
from cubemos.core.nativewrapper import initialise_logging, CM_LogLevel
from cubemos.skeleton_tracking.nativewrapper import Api, SkeletonKeypoints
import cv2
import os
import platform
from pprint import pprint
import glob

import json
from vision_utils.logger import get_logger
logger=get_logger()

keypoint_ids = [
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
]


def default_log_dir():
    if platform.system() == "Windows":
        return os.path.join(os.environ["LOCALAPPDATA"], "Cubemos", "SkeletonTracking", "logs")
    elif platform.system() == "Linux":
        return os.path.join(os.environ["HOME"], ".cubemos", "skeleton_tracking", "logs")
    else:
        raise Exception("{} is not supported".format(platform.system()))


def default_license_dir():
    if platform.system() == "Windows":
        return os.path.join(os.environ["LOCALAPPDATA"], "Cubemos", "SkeletonTracking", "license")
    elif platform.system() == "Linux":
        return os.path.join(os.environ["HOME"], ".cubemos", "skeleton_tracking", "license")
    else:
        raise Exception("{} is not supported".format(platform.system()))


def check_license_and_variables_exist():
    license_path = os.path.join(default_license_dir(), "cubemos_license.json")
    if not os.path.isfile(license_path):
        raise Exception(
            "The license file has not been found at location \"" +
            default_license_dir() + "\". "
            "Please have a look at the Getting Started Guide on how to "
            "use the post-installation script to generate the license file")
    if "CUBEMOS_SKEL_SDK" not in os.environ:
        raise Exception(
            "The environment Variable \"CUBEMOS_SKEL_SDK\" is not set. "
            "Please check the troubleshooting section in the Getting "
            "Started Guide to resolve this issue."
        )


def get_valid_limbs(keypoint_ids, skeleton, confidence_threshold):
    limbs = [
        (tuple(map(int, skeleton.joints[i])), tuple(map(int, skeleton.joints[v])))
        for (i, v) in keypoint_ids
        if skeleton.confidences[i] >= confidence_threshold
        and skeleton.confidences[v] >= confidence_threshold
    ]
    valid_limbs = [
        limb
        for limb in limbs
        if limb[0][0] >= 0 and limb[0][1] >= 0 and limb[1][0] >= 0 and limb[1][1] >= 0
    ]
    return valid_limbs


def render_result(skeletons, img, confidence_threshold):
    skeleton_color = (100, 254, 213)
    for index, skeleton in enumerate(skeletons):
        limbs = get_valid_limbs(keypoint_ids, skeleton, confidence_threshold)
        for limb in limbs:
            cv2.line(
                img, limb[0], limb[1], skeleton_color, thickness=2, lineType=cv2.LINE_AA
            )


import argparse
parser = argparse.ArgumentParser(description='Evaluation options')
parser.add_argument('--scale',
                    default=1,
                    help='Image scaling factor for inference (currently 1 or 0.5)')

args = parser.parse_args()


# Main content begins
if __name__ == "__main__":
    try:

        check_license_and_variables_exist()
        #Get the path of the native libraries and ressource files
        sdk_path = os.environ["CUBEMOS_SKEL_SDK"]

        #initialize the api with a valid license key in default_license_dir()
        api = Api(default_license_dir())
        model_path = os.path.join(
            sdk_path, "models", "skeleton-tracking", "fp32", "skeleton-tracking.cubemos"
        )
        api.load_model(CM_TargetComputeDevice.CM_CPU, model_path)
        #perform inference
        for test_image in glob.glob("/home/slave/Downloads/pose_test_input/*.png"):
            img_name = f'{test_image.split("/")[-1].split(".")[-2]}-{scale}.{test_image.split(".")[-1]}' if scale<1 else test_image.split("/")[-1]
            img = cv2.imread(test_image)
            dim = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            img = cv2.resize(img, dim)
            skeletons = api.estimate_keypoints(img, 192)
            print(skeletons)

            render_result(skeletons, img, 0.5)
            print("Detected skeletons: ", len(skeletons))

            json_out = []
            for person in skeletons:
                keypoints = []
                for i,kp in enumerate(person.joints):
                    keypoints.extend([kp.x, kp.y, person.confidences[i]])
                json_out.append({'keypoints':keypoints})
            print(json_out)

            json_out_name = '../eval/realsense_sdk/' + img_name + '.predictions.json'
            with open(json_out_name, 'w') as f:
                json.dump(json_out, f)
            logger.info(json_out_name)

            cv2.imwrite(img_name, img)



    except Exception as ex:
        print("Exception occured: \"{}\"".format(ex))
# Main content ends
