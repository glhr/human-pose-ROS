#!/usr/bin/env python3

# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse

from utils.img import image_to_numpy, save_image, bgr_to_rgb
from utils.file import get_filename_from_path, get_working_directory
from utils.timing import get_timestamp

def run_openpose(img_path="/home/slave/Downloads/trump.jpg"):
    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            import pyopenpose as op
        except ImportError as e:
            print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e

        # Flags
        parser = argparse.ArgumentParser()
        parser.add_argument("--image_path", default=img_path, help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
        args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "/home/slave/openpose/models/"
        print(params)

        # Add others in path?
        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1])-1: next_item = args[1][i+1]
            else: next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-','')
                if key not in params:  params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-','')
                if key not in params: params[key] = next_item

        # Construct it from system arguments
        # op.init_argv(args[1])
        # oppython = op.OpenposePython()

        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # Process Image
        datum = op.Datum()
        imageToProcess = cv2.imread(args[0].image_path)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])

        # Display Image
        print("Body keypoints: \n" + str(datum.poseKeypoints))
        save_image(bgr_to_rgb(datum.cvOutputData), "{}/test/base_cam_pose_{}.png".format(get_working_directory(), get_timestamp()))
        # cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", datum.cvOutputData)
        # cv2.waitKey(0)
    except Exception as e:
        print(e)
        sys.exit(-1)

if __name__ == "__main__":

    import rospy
    from sensor_msgs.msg import Image
    import numpy as np

    rospy.init_node('human_pose', anonymous=True)
    CAMERA_TOPIC = '/wrist_camera/color/image_raw'

    imagemsg = rospy.wait_for_message(CAMERA_TOPIC, Image)
    image = image_to_numpy(imagemsg)
    print("saving image")
    save_image(image, get_working_directory()+"/test/base_cam_{}.png".format(get_timestamp()))

    run_openpose(get_working_directory()+"/test/base_cam.png")
    sys.exit(0)
