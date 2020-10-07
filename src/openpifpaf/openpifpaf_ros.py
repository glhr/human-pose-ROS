#!/usr/bin/env python3
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

from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from human_pose_ROS.msg import Skeleton, PoseEstimation

from vision_utils.img import image_to_numpy, numpy_to_image, load_image
from vision_utils.logger import get_logger, get_printer
from vision_utils.timing import get_timestamp
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


def predict(img_path, scale=1, json_output=None, save=True):
    if isinstance(img_path, str):
        pil_im = PIL.Image.open(img_path)
        img_name = img_path.split("/")[-1]
        img_name = f'{img_path.split("/")[-1].split(".")[-2]}-{scale}.{img_path.split(".")[-1]}' if scale<1 else img_path.split("/")[-1]
    else:
        pil_im = PIL.Image.fromarray(img_path)
        img_name = f'wrist_camera_{get_timestamp()}.png'
    dim = (int(i*scale) for i in pil_im.size)
    pil_im = pil_im.resize(dim)

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

    logger.info(f"{img_name} took {timer.took}ms")

    for predictions in predictions_list:
      with openpifpaf.show.image_canvas(im, f"out/{img_name}" if save else None, show=False) as ax:
        keypoint_painter.annotations(ax, predictions)


    return predictions_list, load_image(f"out/{img_name}")[:,:,:3], timer.took



parser = argparse.ArgumentParser(description='Directory of PNG images to use for inference.')
parser.add_argument('--input_dir',
                    default="/home/slave/Pictures/pose/pose test input",
                    help='directory of PNG images to run fastpose on')
parser.add_argument('--cam', dest='cam', action='store_true')

args = parser.parse_args()

img_path = "/home/robotlab/pose test input/wrist_cam_1600951547.png"

pairs = dict(list(enumerate(openpifpaf.datasets.constants.COCO_KEYPOINTS)))
pp.pprint(pairs)

RGB_CAMERA_TOPIC = '/wrist_camera/camera/color/image_raw'
DEPTH_CAMERA_TOPIC = '/wrist_camera/camera/aligned_depth_to_color/image_raw'
DEPTH_INFO_TOPIC = '/wrist_camera/camera/aligned_depth_to_color/camera_info'
depth_image = []
rgb_image = []
def got_depth(msg):
    global depth_image
    depth_image = image_to_numpy(msg)
def got_rgb(msg):
    global rgb_image
    rgb_image = image_to_numpy(msg)

skel_pub = rospy.Publisher('openpifpaf_markers', Marker, queue_size=100)
pose_pub = rospy.Publisher('openpifpaf_pose', PoseEstimation, queue_size=1)
poseimg_pub = rospy.Publisher('openpifpaf_img', Image, queue_size=1)
depth_sub = rospy.Subscriber(DEPTH_CAMERA_TOPIC, Image, got_depth)
rgb_sub = rospy.Subscriber(RGB_CAMERA_TOPIC, Image, got_rgb)
rospy.init_node('openpifpaf')

colors = dict()
for k in range(10):
  colors[k] = tuple(np.random.randint(256, size=3)/256)

connected_points = [
(0,2), (2,4), (4,6), (6,8), (8,10),
(0,1), (1,3), (3,5), (5,7), (7,9),
(5,11), (11,13), (13,15),
(6,12), (12,14), (14,16),
(11,12), (5,6)]

def get_points_centroid(arr):
    length = len(arr)
    arr = np.array(arr)
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    sum_z = np.sum(arr[:, 2])
    return sum_x/length, sum_y/length, sum_z/length

def angle_from_centroid(centroid):
    v0 = np.array([0,0,1])
    vcentroid = np.array(centroid)
    angle = vg.angle(v0, vcentroid)
    logger.debug(f"Centroid angle: {angle}")
    return angle

def skeleton_from_keypoints(skel_dict):
    skel = Skeleton()
    skel = skel_dict
    pp.pprint(skel)
    return message_converter.convert_dictionary_to_ros_message('human_pose_ROS/Skeleton', skel)

def openpifpaf_viz(predictions, im, time, cam=True, scale=1):
    predictions = [ann.json_data() for ann in predictions[0]]
    pose_msg = PoseEstimation()
    pose_msg.skeletons = []

    for person_id, person in enumerate(predictions):

        pnts_openpifpaf = person['keypoints']
        pnts_openpifpaf = list(zip(pnts_openpifpaf[0::3], pnts_openpifpaf[1::3]))
        #d = dict(zip(keys, values))
        pnts_dict = dict()

        skel_dict = dict()

        for i,conn in enumerate(connected_points):
            pnt_1 = tuple(pnt/scale for pnt in pnts_openpifpaf[conn[0]])
            pnt_2 = tuple(pnt/scale for pnt in pnts_openpifpaf[conn[1]])
            if pnt_1[0] > 0 and pnt_1[1] > 0 and pnt_2[0] > 0 and pnt_2[1] > 0:

                if cam:
                    pnt1_cam = pixel_to_camera(pnt_1, depth_image[int(pnt_1[1])][int(pnt_1[0])]/1000)
                    pnt2_cam = pixel_to_camera(pnt_2, depth_image[int(pnt_2[1])][int(pnt_2[0])]/1000)
                else:
                    pnt1_cam = [i/100 for i in pnt_1]
                    pnt2_cam = [i/100 for i in pnt_2]
                    pnt1_cam.append(1)
                    pnt2_cam.append(1)

                skel_dict[pairs[conn[0]]] = pnt1_cam
                skel_dict[pairs[conn[1]]] = pnt2_cam

                now = rospy.get_rostime()
                line_marker = Marker()
                line_marker.header.frame_id = "/wrist_camera_link" if cam else "/map"
                line_marker.type = line_marker.LINE_STRIP
                line_marker.action = line_marker.ADD
                line_marker.scale.x = 0.02
                line_marker.color.a = 1.0
                line_marker.color.r, line_marker.color.g, line_marker.color.b = colors[person_id]
                line_marker.pose.orientation.w = 1.0
                line_marker.pose.position.x = 0
                line_marker.pose.position.y = 0
                line_marker.pose.position.z = 0
                line_marker.id = person_id*100 + i*2+1
                line_marker.lifetime = rospy.Duration(time*4/1000)
                line_marker.header.stamp = now
                line_marker.points = []
                # first point
                first_line_point = Point()
                first_line_point.x = pnt1_cam[0]
                first_line_point.y = pnt1_cam[1]
                first_line_point.z = pnt1_cam[2]
                line_marker.points.append(first_line_point)
                # second point
                second_line_point = Point()
                second_line_point.x = pnt2_cam[0]
                second_line_point.y = pnt2_cam[1]
                second_line_point.z = pnt2_cam[2]
                line_marker.points.append(second_line_point)

                pp.pprint(line_marker.points)
                skel_pub.publish(line_marker)

                pnt_marker = Marker()
                pnt_marker.header.frame_id = "/wrist_camera_link" if cam else "/map"
                pnt_marker.type = pnt_marker.SPHERE
                pnt_marker.action = pnt_marker.ADD
                pnt_marker.scale.x, pnt_marker.scale.y, pnt_marker.scale.z = 0.03, 0.03, 0.03
                pnt_marker.color.a = 1.0
                pnt_marker.color.r, pnt_marker.color.g, pnt_marker.color.b = (1.0,1.0,1.0)
                pnt_marker.pose.orientation.w = 1.0
                pnt_marker.pose.position.x = pnt1_cam[0]
                pnt_marker.pose.position.y = pnt1_cam[1]
                pnt_marker.pose.position.z = pnt1_cam[2]
                # logger.debug(pnt_marker.pose.position)
                pnt_marker.id = person_id*100 + i*2
                pnt_marker.lifetime = rospy.Duration(time*4/1000)
                pnt_marker.header.stamp = now
                skel_pub.publish(pnt_marker)
                pnt_marker.pose.position.x = pnt2_cam[0]
                pnt_marker.pose.position.y = pnt2_cam[1]
                pnt_marker.pose.position.z = pnt2_cam[2]
                skel_pub.publish(pnt_marker)

        skeleton_msg = skeleton_from_keypoints(skel_dict)
        skel_centroid = get_points_centroid(list(skel_dict.values()))
        logger.info(f"Centroid: {skel_centroid}")
        centroid_marker = Marker()
        centroid_marker.header.frame_id = "/wrist_camera_link" if cam else "/map"
        centroid_marker.type = centroid_marker.SPHERE
        centroid_marker.action = centroid_marker.ADD
        centroid_marker.scale.x, centroid_marker.scale.y, centroid_marker.scale.z = 0.1, 0.1, 0.1
        centroid_marker.color.a = 1.0
        centroid_marker.color.r, centroid_marker.color.g, centroid_marker.color.b = (1.0,0.0,0.0)
        centroid_marker.pose.orientation.w = 1.0
        centroid_marker.pose.position.x = skel_centroid[0]
        centroid_marker.pose.position.y = skel_centroid[1]
        centroid_marker.pose.position.z = skel_centroid[2]
        centroid_marker.id = person_id
        centroid_marker.lifetime = rospy.Duration(time*4/1000)
        centroid_marker.header.stamp = now
        skeleton_msg.centroid = skel_centroid
        skel_pub.publish(centroid_marker)

        angle_from_centroid(skel_centroid)

        pose_msg.skeletons.append(skeleton_msg)

    poseimg_pub.publish(numpy_to_image(im))
    pose_pub.publish(pose_msg)

if args.cam:
    cameraInfo = rospy.wait_for_message(DEPTH_INFO_TOPIC, CameraInfo, timeout=2)
    logger.info("Got camera info")

def pixel_to_camera(pixel, depth):
    _intrinsics = rs.intrinsics()
    _intrinsics.width = cameraInfo.width
    _intrinsics.height = cameraInfo.height
    _intrinsics.ppx = cameraInfo.K[2]
    _intrinsics.ppy = cameraInfo.K[5]
    _intrinsics.fx = cameraInfo.K[0]
    _intrinsics.fy = cameraInfo.K[4]
    #_intrinsics.model = cameraInfo.distortion_model
    _intrinsics.model  = rs.distortion.none
    _intrinsics.coeffs = [i for i in cameraInfo.D]
    x,y = pixel
    result = rs.rs2_deproject_pixel_to_point(_intrinsics, [x,y], depth)
    # return rs.rs2_deproject_pixel_to_point(depth_intrin, pixel, 1.0)
    return result

# rosrun tf static_transform_publisher 0 0 0 0 0 0 1 map my_frame 10
imgs = list(glob.glob(f"{args.input_dir}/*.png"))
while not rospy.is_shutdown():
    if not args.cam:
        img_path = imgs[randint(0,len(imgs)-1)]
        # img_path = "/home/robotlab/pose test input/wrist_cam_1600951547.png"
        predictions, im, time = predict(img_path, scale=0.5)
        openpifpaf_viz(predictions, im, time, cam=False)
    else:
        # imagemsg = rospy.wait_for_message(RGB_CAMERA_TOPIC, Image, timeout=2)
        # depth_imagemsg = rospy.wait_for_message(DEPTH_CAMERA_TOPIC, Image, timeout=2)
        # image = image_to_numpy(imagemsg)
        if len(depth_image):
            predictions, im, time = predict(rgb_image, scale=0.5)
            openpifpaf_viz(predictions, im, time, cam=True, scale=0.5)
            # pp.pprint(markerArray.markers)
            # pp.pprint(pnts_dict)
# while True:
#     predictions, time = predict(img_path, scale=0.5)
#     times.append(time)
# print(f"Inference took {np.mean(times)}ms per image (avg)" )
