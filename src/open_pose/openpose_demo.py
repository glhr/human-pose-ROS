#!/usr/bin/env python3

# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse

from vision_utils.img import image_to_numpy, save_image, bgr_to_rgb
from vision_utils.file import get_filename_from_path, get_working_directory
from vision_utils.timing import get_timestamp
from vision_utils.timing import CodeTimer
from vision_utils.logger import get_logger
logger = get_logger()
import json

def run_openpose(img_path="/home/slave/Downloads/trump.jpg", scale=1):
    img_name = f'{img_path.split("/")[-1].split(".")[-2]}-{scale}.{img_path.split(".")[-1]}' if scale<1 else img_path.split("/")[-1]
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
    # params["number_people_max"] = 1
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
    with CodeTimer() as timer:
        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # Process Image
        datum = op.Datum()
        imageToProcess = cv2.imread(args[0].image_path)
        dim = (int(imageToProcess.shape[1] * scale), int(imageToProcess.shape[0] * scale))
        imageToProcess = cv2.resize(imageToProcess, dim)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])
    print(img_name, timer.took)

    # Display Image
    json_out = []
    print("Body keypoints: \n" + str(datum.poseKeypoints))
    keypoints = datum.poseKeypoints.flatten().tolist()
    json_out.append({'keypoints':keypoints})        
    json_out_name = '../eval/openpose/' + img_name + '.predictions.json'
    with open(json_out_name, 'w') as f:
        json.dump(json_out, f)
    logger.info(json_out_name)
    save_image(bgr_to_rgb(datum.cvOutputData), img_name)
    return timer.took
    # cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", datum.cvOutputData)
    # cv2.waitKey(0)

if __name__ == "__main__":

    import rospy
    from sensor_msgs.msg import Image
    import numpy as np
    import rosgraph

    CAMERA_TOPIC = '/wrist_camera/color/image_raw'

    if rosgraph.is_master_online():
        rospy.init_node('human_pose', anonymous=True)
        imagemsg = rospy.wait_for_message(CAMERA_TOPIC, Image, timeout=2)
        image = image_to_numpy(imagemsg)
        print("saving image")
        timestamp = get_timestamp()
        save_image(image, get_working_directory()+"/test/base_cam_{}.png".format(timestamp))
        img_path = get_working_directory()+"/test/base_cam_{}.png".format(timestamp)
        run_openpose(img_path)
    else:
        import glob
        import numpy
        print("loading images from file")
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--input_dir',
                            default="/home/slave/Pictures/pose/pose test input",
                            help='directory of PNG images to run fastpose on')

        args = parser.parse_args()
        times = []
        for test_image in glob.glob(f"{args.input_dir}/*.png"):
            time = run_openpose(test_image, scale=1)
            times.append(time)

        print(np.mean(times))

    sys.exit(0)
