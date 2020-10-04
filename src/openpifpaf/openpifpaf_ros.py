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
print(pairs)

skel_pub = rospy.Publisher('openpifpaf_skeleton', MarkerArray, queue_size=100)
img_pub = rospy.Publisher('openpifpaf_img', Image, queue_size=10)
rospy.init_node('openpifpaf')

colors = dict()
for k in range(10):
  colors[k] = tuple(np.random.randint(256, size=3)/256)

# rosrun tf static_transform_publisher 0 0 0 0 0 0 1 map my_frame 10
while not rospy.is_shutdown():

    for img_path in glob.glob(f"{args.input_dir}/*.png"):
        predictions, im, time = predict(img_path, scale=0.5)
        predictions = [ann.json_data() for ann in predictions[0]]
        img_pub.publish(numpy_to_image(im))

        markerArray = MarkerArray()
        for person_id, person in enumerate(predictions):

            pnts_openpifpaf = person['keypoints']
            pnts_openpifpaf = list(zip(pnts_openpifpaf[0::3], pnts_openpifpaf[1::3]))
            #d = dict(zip(keys, values))
            pnts_dict = dict()
            for i,pnt in enumerate(pnts_openpifpaf):
                pnts_dict[pairs[i]] = pnt

                if pnt[0] > 0 and pnt[1] > 0:
                    marker = Marker()
                    marker.header.frame_id = "/map"
                    marker.type = marker.SPHERE
                    marker.action = marker.ADD
                    marker.scale.x = 0.1
                    marker.scale.y = 0.1
                    marker.scale.z = 0.1
                    marker.color.a = 1.0
                    marker.color.r, marker.color.g, marker.color.b = colors[person_id]
                    marker.pose.orientation.w = 1.0
                    marker.pose.position.x = pnt[0]/100
                    marker.pose.position.y = pnt[1]/100
                    marker.pose.position.z = 1
                    marker.id = i + person_id*10
                    marker.lifetime = rospy.Duration(0.5)
                    marker.header.stamp = rospy.get_rostime()
                    markerArray.markers.append(marker)
                    # pp.pprint([m.id for m in markerArray.markers])


        skel_pub.publish(markerArray)
            # pp.pprint(markerArray.markers)
            # pp.pprint(pnts_dict)

# while True:
#     predictions, time = predict(img_path, scale=0.5)
#     times.append(time)
# print(f"Inference took {np.mean(times)}ms per image (avg)" )
