#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from vision_utils.img import image_to_numpy, save_image, load_image, rgb_to_bgr
from vision_utils.file import get_filename_from_path, get_working_directory
from vision_utils.timing import get_timestamp
import numpy as np

camera = 'wrist'
CAMERA_TOPIC_DEPTH = "/wrist_camera/camera/aligned_depth_to_color/image_raw"

images = dict()
store_images = True
i = 0

def depth_cb(message):
    global i
    i += 1
    if store_images:
        images[i] = message
        print(i)

rospy.init_node('get_image', anonymous=True)

distances_sub = rospy.Subscriber(CAMERA_TOPIC_DEPTH, Image, depth_cb)

while not input():
    pass

store_images = False

for t,image in images.items():
    image = image_to_numpy(image)
    image = image / np.max(image)
    save_image(image, f"out/depth_cam_{t}.png")
