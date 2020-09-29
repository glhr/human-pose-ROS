#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from vision_utils.img import image_to_numpy, save_image, load_image, rgb_to_bgr
from vision_utils.file import get_filename_from_path, get_working_directory
from vision_utils.timing import get_timestamp

camera = 'wrist'
CAMERA_TOPIC = f'/{camera}_camera/camera/color/image_raw'

rospy.init_node('human_pose', anonymous=True)

while(not len(input())):
    imagemsg = rospy.wait_for_message(CAMERA_TOPIC, Image)
    print("got image")
    image = image_to_numpy(imagemsg)

    timestamp = get_timestamp()
    save_image(image, f"test/{camera}_cam_{timestamp}.png")
