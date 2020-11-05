#!/usr/bin/env python3
import sys
import io
import numpy as np
import openpifpaf
import PIL
import requests
import torch

import pyrealsense2 as rs

import matplotlib
# matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt
from random import randint
import glob
import numpy as np
import vg
import argparse
times = []
import roslib
import rospy
from rospy_message_converter import message_converter
from scipy.ndimage import gaussian_filter
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float

from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from human_pose_ROS.msg import Skeleton, PoseEstimation
from pose_utils.utils import pixel_to_camera, get_points_centroid, angle_from_centroid

from vision_utils.img import image_to_numpy, numpy_to_image, load_image
from vision_utils.logger import get_logger, get_printer
from vision_utils.timing import get_timestamp

import pathlib
filepath = pathlib.Path(__file__).parent.absolute()

logger = get_logger()
pp = get_printer()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(openpifpaf.__version__)
print(torch.__version__)

from vision_utils.timing import CodeTimer

net_cpu, _ = openpifpaf.network.factory(checkpoint='shufflenetv2k16w', download_progress=False)
net = net_cpu.to(device)

openpifpaf.decoder.CifSeeds.threshold = 0.5
openpifpaf.decoder.nms.Keypoints.keypoint_threshold = 0.2
openpifpaf.decoder.nms.Keypoints.instance_threshold = 0.2
processor = openpifpaf.decoder.factory_decode(net.head_nets, basenet_stride=net.base_net.stride)

preprocess = openpifpaf.transforms.Compose([
    openpifpaf.transforms.NormalizeAnnotations(),
    openpifpaf.transforms.CenterPadTight(16),
    openpifpaf.transforms.EVAL_TRANSFORM,
])

depth_image = []
depth_predict = []
rgb_image = []

parser = argparse.ArgumentParser(description='Directory of PNG images to use for inference.')
parser.add_argument('--input_dir',
                    default="/home/slave/Pictures/pose/pose test input",
                    help='directory of PNG images to run fastpose on')
parser.add_argument('--cam', dest='cam', action='store_true', default=False)
parser.add_argument('--webcam', dest='webcam', action='store_true')
parser.add_argument('--save', dest='save', action='store_true')
parser.add_argument('--scale', default=0.5, dest='scale')
parser.add_argument('--realsense', default='wrist')
parser.add_argument('--debug',
                default=False,
                 action='store_true',
                 help='Print openpifpaf debug')

args, unknown = parser.parse_known_args()

aspect_ratio = 480/848

valid_scales = {0.25,0.5,0.75}

def predict(img_path, scale=1, json_output=None):

    if isinstance(img_path, str):
        pil_im = PIL.Image.open(img_path)
        img_name = img_path.split("/")[-1]
        img_name = f'{img_path.split("/")[-1].split(".")[-2]}-{scale}.{img_path.split(".")[-1]}' if scale<1 else img_path.split("/")[-1]
    else:
        pil_im = PIL.Image.fromarray(np.array(img_path))
        if args.cam:
            pil_im_depth = PIL.Image.fromarray(depth_predict)
        img_name = f'{args.realsense}_camera_{get_timestamp()}.png'

    if scale < 1 and scale in valid_scales:
        dim = (round(i*scale) for i in pil_im.size)
        pil_im = pil_im.resize(dim)
    elif scale < 1:
        logger.warning("Scale should be in {valid_scales}, not resizing")
    if args.cam:
        if scale < 1 and scale in valid_scales:
            dim = (round(i*scale) for i in pil_im_depth.size)
            pil_im_depth = pil_im_depth.convert('F')
            pil_im_depth = pil_im_depth.resize(dim)
        elif scale < 1:
            logger.warning("Scale should be in {valid_scales}, not resizing")
        im_depth = np.asarray(pil_im_depth)
        # im_depth = gaussian_filter(im_depth, sigma=2)
        # im_depth = denoise_tv_chambolle(im_depth, multichannel=False, weight=0.2)
        # poseimg_pub.publish(numpy_to_image(im_depth, encoding="32FC1"))

    im = np.asarray(pil_im)

    predictions_list = []

    with CodeTimer() as timer:
        data = openpifpaf.datasets.PilImageList([pil_im], preprocess=preprocess)

        loader = torch.utils.data.DataLoader(
          data, batch_size=1, pin_memory=True,
          collate_fn=openpifpaf.datasets.collate_images_anns_meta)

        keypoint_painter = openpifpaf.show.KeypointPainter(color_connections=True, linewidth=6)

        for images_batch, _, __ in loader:
            predictions = processor.batch(net, images_batch, device=device)[0]
            predictions_list.append(predictions)

            if json_output is not None:
                json_out_name = json_output + '/' + img_name + '.predictions.json'
                logger.debug('json output = %s', json_out_name)
                with open(json_out_name, 'w') as f:
                    json.dump([ann.json_data() for ann in predictions], f)

    if args.debug: logger.info(f"{img_name} took {timer.took}ms")

    if im is not None:
        try:
            poseimg_pub.publish(numpy_to_image(im, encoding="rgb8"))
        except TypeError:
            poseimg_pub.publish(numpy_to_image(im, encoding="32FC1"))

    if args.save:
        save_path_depth = f"{filepath}/out_depth/depth-{img_name}"
        save_path_rgb = f"{filepath}/out_rgb/rgb-{img_name}"
        save_path = save_path_depth if args.cam else save_path_rgb
        im = im_depth if args.cam else im
        if args.debug: logger.warning("Saving to {}".format(save_path))
        try:
            for predictions in predictions_list:
              # with openpifpaf.show.image_canvas(im, f"out/{img_name}" if save else None, show=False) as ax:
              #   keypoint_painter.annotations(ax, predictions)
                with openpifpaf.show.image_canvas(im, save_path, show=False) as ax:
                    keypoint_painter.annotations(ax, predictions)
                image = load_image(save_path)[:,:,:3]
                save_pub.publish(save_path)
        except Exception as e:
            image = np.zeros_like(im)

        return predictions_list, image, timer.took

    return predictions_list, np.zeros_like(im), timer.took


img_path = "/home/robotlab/pose test input/wrist_cam_1600951547.png"

pairs = dict(list(enumerate(openpifpaf.datasets.constants.COCO_KEYPOINTS)))
pp.pprint(pairs)

RGB_CAMERA_TOPIC = f'/{args.realsense}_camera/camera/color/image_raw'
DEPTH_CAMERA_TOPIC = f'/{args.realsense}_camera/camera/aligned_depth_to_color/image_raw'
DEPTH_INFO_TOPIC = f'/{args.realsense}_camera/camera/aligned_depth_to_color/camera_info'

def got_depth(msg):
    global depth_image
    depth_image = image_to_numpy(msg)
def got_rgb(msg):
    global rgb_image
    rgb_image = image_to_numpy(msg)

pose_pub = rospy.Publisher('openpifpaf_pose', PoseEstimation, queue_size=1)
save_pub = rospy.Publisher('openpifpaf_savepath', String, queue_size=1)
poseimg_pub = rospy.Publisher('openpifpaf_img', Image, queue_size=1)
# angle_pub = rospy.Publisher('person_angle', Float32, queue_size=1)
depth_sub = rospy.Subscriber(DEPTH_CAMERA_TOPIC, Image, got_depth)
rgb_sub = rospy.Subscriber(RGB_CAMERA_TOPIC, Image, got_rgb)
rospy.init_node('openpifpaf')

def skeleton_from_keypoints(skel_dict):
    skel = skel_dict
    if args.debug: pp.pprint(skel)
    return message_converter.convert_dictionary_to_ros_message('human_pose_ROS/Skeleton', skel_dict)

depth_history = dict()


def openpifpaf_viz(predictions, im, time, cam=True, scale=1):
    predictions = [ann.json_data() for ann in predictions[0]]
    pose_msg = PoseEstimation()
    pose_msg.skeletons = []

    im_w = int(im.shape[1]/scale)
    im_h = int(im.shape[0]/scale)

    angles = dict()
    for person_id, person in enumerate(predictions):

        if not depth_history.get(person_id,0):
            depth_history[person_id] = dict()
        pnts_openpifpaf = person['keypoints']
        pnts_openpifpaf = list(zip(pnts_openpifpaf[0::3], pnts_openpifpaf[1::3]))
        #d = dict(zip(keys, values))
        pnts_dict = dict()

        skel_dict = dict()

        for i,pnt in enumerate(pnts_openpifpaf):
            pnt_1 = tuple(pnt/scale for pnt in pnts_openpifpaf[i])
            # pnt_1 = pnts_openpifpaf[i]

            if pnt_1[0] > 0 and pnt_1[1] > 0:
                if cam:
                    if pnt_1[1] >= im_h-10 or pnt_1[0] >= im_w-10:
                        if args.debug: logger.error(pnt_1)
                    y = min(im_h-1, int(pnt_1[1]))
                    x = min(im_w-1, int(pnt_1[0]))
                    z = depth_predict[y if y<im_h-1 else im_h-10][x if y<im_w-1 else im_w-10]/1000
                    pnt1_cam = [x,y,z]
                else:
                    pnt1_cam = [i/100 for i in pnt_1]
                    pnt1_cam.append(1)

                # if pnt1_cam[2] > 0.1 and (not depth_history[person_id].get(i,0) or abs(depth_history[person_id][i][2] - pnt1_cam[2])<1.0):
                skel_dict[pairs[i]] = pnt1_cam

                    # depth_history[person_id][i] = pnt1_cam


        skeleton_msg = skeleton_from_keypoints(skel_dict)

        # angle = angle_from_centroid(skel_centroid)
        # angles[person_id] = 0
        # logger.info(angle)
        # angle_pub.publish(angle)

        pose_msg.skeletons.append(skeleton_msg)


    pose_pub.publish(pose_msg)

if not args.webcam:
    cameraInfo = rospy.wait_for_message(DEPTH_INFO_TOPIC, CameraInfo, timeout=2)
    logger.info("Got camera info")

# rosrun tf static_transform_publisher 0 0 0 0 0 0 1 map my_frame 10
imgs = list(glob.glob(f"{args.input_dir}/*.png"))
if args.webcam:
    import cv2
    cap = cv2.VideoCapture(0)

while not rospy.is_shutdown():
    if args.webcam:
        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictions, im, time = predict(img, scale=float(args.scale))
        openpifpaf_viz(predictions, im, time, cam=False)
    elif not args.cam:
        img_path = imgs[randint(0,len(imgs)-1)]
        # img_path = "/home/robotlab/pose test input/base_cam_1600951547.png"
        predictions, im, time = predict(img_path, scale=float(args.scale))
        openpifpaf_viz(predictions, im, time, cam=False)
    else:
        # imagemsg = rospy.wait_for_message(RGB_CAMERA_TOPIC, Image, timeout=2)
        # depth_imagemsg = rospy.wait_for_message(DEPTH_CAMERA_TOPIC, Image, timeout=2)
        # image = image_to_numpy(imagemsg)
        if len(depth_image):
            depth_predict = depth_image
            predictions, im, time = predict(rgb_image, scale=float(args.scale))
            openpifpaf_viz(predictions, im, time, cam=True, scale=float(args.scale))
            # pp.pprint(markerArray.markers)
            # pp.pprint(pnts_dict)
# while True:
#     predictions, time = predict(img_path, scale=0.5)
#     times.append(time)
# print(f"Inference took {np.mean(times)}ms per image (avg)" )
