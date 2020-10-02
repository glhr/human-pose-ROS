#!/usr/bin/env python3
import io
import numpy as np
import openpifpaf
import PIL
import requests
import torch
import json

import matplotlib
matplotlib.use('gtk3agg')
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(openpifpaf.__version__)
print(torch.__version__)

from vision_utils.timing import CodeTimer
from vision_utils.logger import get_logger

logger = get_logger()

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

    print(img_name, timer.took)

    for predictions in predictions_list:
      with openpifpaf.show.image_canvas(im, img_name, show=False) as ax:
        keypoint_painter.annotations(ax, predictions)

    return predictions, timer.took

import glob
import numpy as np
import argparse
times = []

parser = argparse.ArgumentParser(description='Directory of PNG images to use for inference.')
parser.add_argument('--input_dir',
                    default="/home/slave/Pictures/pose/pose test input",
                    help='directory of PNG images to run fastpose on')
parser.add_argument('--json_output',
                    default="../eval/openpifpaf",
                    help='JSON output directory')
parser.add_argument('--scale',
                    default=1,
                    help='JSON output directory')

args = parser.parse_args()

for img_path in glob.glob(f"{args.input_dir}/*.png"):
    _, time = predict(img_path, scale=args.scale, json_output=args.json_output)
    times.append(time)
print(f"Inference took {np.mean(times)}ms per image (avg)" )
