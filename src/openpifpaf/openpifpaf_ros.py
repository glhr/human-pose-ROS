#!/usr/bin/env python3
import io
import numpy as np
import openpifpaf
import PIL
import requests
import torch

import matplotlib
# matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt

import glob
import numpy as np
import argparse
times = []
import roslib
import rospy

from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from vision_utils.img import numpy_to_image, load_image
from vision_utils.logger import get_logger, get_printer
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

def predict(img_path, scale=1, json_output=None):
    pil_im = PIL.Image.open(img_path)
    dim = (int(i*scale) for i in pil_im.size)
    pil_im = pil_im.resize(dim)

    img_name = img_path.split("/")[-1]
    img_name = f'{img_path.split("/")[-1].split(".")[-2]}-{scale}.{img_path.split(".")[-1]}' if scale<1 else img_path.split("/")[-1]
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
      with openpifpaf.show.image_canvas(im, img_name, show=False) as ax:
        keypoint_painter.annotations(ax, predictions)


    return predictions_list, load_image(img_name)[:,:,:3], timer.took



parser = argparse.ArgumentParser(description='Directory of PNG images to use for inference.')
parser.add_argument('--input_dir',
                    default="/home/slave/Pictures/pose/pose test input",
                    help='directory of PNG images to run fastpose on')

args = parser.parse_args()

img_path = "/home/robotlab/pose test input/wrist_cam_1600951547.png"

pairs = dict(list(enumerate(openpifpaf.datasets.constants.COCO_KEYPOINTS)))
pp.pprint(pairs)

skel_pub = rospy.Publisher('openpifpaf_skeleton', Marker, queue_size=100)
img_pub = rospy.Publisher('openpifpaf_img', Image, queue_size=10)
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

# rosrun tf static_transform_publisher 0 0 0 0 0 0 1 map my_frame 10
while not rospy.is_shutdown():

    for img_path in glob.glob(f"{args.input_dir}/*.png"):
        # img_path = "/home/robotlab/pose test input/wrist_cam_1600951547.png"
        predictions, im, time = predict(img_path, scale=0.5)
        predictions = [ann.json_data() for ann in predictions[0]]
        img_pub.publish(numpy_to_image(im))

        for person_id, person in enumerate(predictions):

            pnts_openpifpaf = person['keypoints']
            pnts_openpifpaf = list(zip(pnts_openpifpaf[0::3], pnts_openpifpaf[1::3]))
            #d = dict(zip(keys, values))
            pnts_dict = dict()

            for i,conn in enumerate(connected_points):
                pnt_1 = pnts_openpifpaf[conn[0]]
                pnt_2 = pnts_openpifpaf[conn[1]]
                if pnt_1[0] > 0 and pnt_1[1] > 0 and pnt_2[0] > 0 and pnt_2[1] > 0:
                    time = rospy.get_rostime()
                    line_marker = Marker()
                    line_marker.header.frame_id = "/map"
                    line_marker.type = line_marker.LINE_STRIP
                    line_marker.action = line_marker.ADD
                    line_marker.scale.x, line_marker.scale.y, line_marker.scale.z = 0.05, 0.05, 0.05
                    line_marker.color.a = 1.0
                    line_marker.color.r, line_marker.color.g, line_marker.color.b = colors[person_id]
                    line_marker.pose.orientation.w = 1.0
                    line_marker.pose.position.x = 0
                    line_marker.pose.position.y = 0
                    line_marker.pose.position.z = 1
                    line_marker.id = person_id*100 + i*2+1
                    line_marker.lifetime = rospy.Duration(1)
                    line_marker.header.stamp = time
                    line_marker.points = []
                    # first point
                    first_line_point = Point()
                    first_line_point.x = pnt_1[0]/100
                    first_line_point.y = pnt_1[1]/100
                    first_line_point.z = 0.0
                    line_marker.points.append(first_line_point)
                    # second point
                    second_line_point = Point()
                    second_line_point.x = pnt_2[0]/100
                    second_line_point.y = pnt_2[1]/100
                    second_line_point.z = 0.0
                    line_marker.points.append(second_line_point)

                    # pp.pprint(marker.points)

                    skel_pub.publish(line_marker)

                    pnt_marker = Marker()
                    pnt_marker.header.frame_id = "/map"
                    pnt_marker.type = pnt_marker.SPHERE
                    pnt_marker.action = pnt_marker.ADD
                    pnt_marker.scale.x, pnt_marker.scale.y, pnt_marker.scale.z = 0.1, 0.1, 0.1
                    pnt_marker.color.a = 1.0
                    pnt_marker.color.r, pnt_marker.color.g, pnt_marker.color.b = (1.0,1.0,1.0)
                    pnt_marker.pose.orientation.w = 1.0
                    pnt_marker.pose.position.x = pnt_1[0]/100
                    pnt_marker.pose.position.y = pnt_1[1]/100
                    pnt_marker.pose.position.z = 1
                    pnt_marker.id = person_id*100 + i*2
                    pnt_marker.lifetime = rospy.Duration(1)
                    pnt_marker.header.stamp = time
                    skel_pub.publish(pnt_marker)
        # pp.pprint(markerArray.markers)
        # pp.pprint(pnts_dict)

# while True:
#     predictions, time = predict(img_path, scale=0.5)
#     times.append(time)
# print(f"Inference took {np.mean(times)}ms per image (avg)" )
