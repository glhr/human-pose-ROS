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

from vision_utils.timing import CodeTimer
from vision_utils.logger import get_logger
from random import randint

logger = get_logger()

def predict(img_path, scale=1, json_output=None):
    scale = float(scale)
    pil_im = PIL.Image.open(img_path)
    dim = list(int(i*scale) for i in pil_im.size)
    pil_im = pil_im.resize(dim)

    img_name = img_path.split("/")[-1]
    img_name = f'{img_path.split("/")[-1].split(".")[-2]}-{scale}.{img_path.split(".")[-1]}' if scale<1 else img_path.split("/")[-1]
    im = np.asarray(pil_im)

    with CodeTimer() as timer:
        predictions_dict = {'keypoints': []}

        for p in range(18):
            predictions_dict['keypoints'].extend([randint(0,dim[0]), randint(0,dim[1]), 1])

        if json_output is not None:
            json_out_name = json_output + '/' + img_name + '.predictions.json'
            logger.debug('json output = %s', json_out_name)
        with open(json_out_name, 'w') as f:
            json.dump([predictions_dict], f)

    print(img_name, timer.took)

    return predictions_dict, timer.took

import glob
import numpy as np
import argparse
times = []

parser = argparse.ArgumentParser(description='Directory of PNG images to use for inference.')
parser.add_argument('--input_dir',
                    default="/home/slave/Pictures/pose/pose test input",
                    help='directory of PNG images to run fastpose on')
parser.add_argument('--json_output',
                    default="../eval/random",
                    help='JSON output directory')
parser.add_argument('--scale',
                    default=1,
                    help='JSON output directory')

args = parser.parse_args()

for img_path in glob.glob(f"{args.input_dir}/*.png"):
    _, time = predict(img_path, scale=args.scale, json_output=args.json_output)
    times.append(time)
print(f"Inference took {np.mean(times)}ms per image (avg)" )
