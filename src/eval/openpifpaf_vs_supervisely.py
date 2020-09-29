##
# Author: Caio Marcellos
# Email: caiocuritiba@gmail.com
##
import os
import numpy as np
import json
import glob
from datetime import datetime
from pathlib import Path
import argparse
import sys



mapping_ann_openpifpaf = {
  0: 0,
  1: 17,
  2: 5,
  3: 7,
  4: 9,
  5: 6,
  6: 8,
  7: 10,
  8: 11,
  9: 13,
  10: 15,
  11: 12,
  12: 14,
  13: 16,
  14: 1,
  15: 2,
  16: 3,
  17: 4
}

## process annotation metadata

kp_labels = dict()

with open('meta.json') as f:
  meta = json.load(f)

for cls in meta["classes"]:
  if not cls["title"] == "pose":
    continue
  kps = cls["geometry_config"]["nodes"]
  for kp in kps.items():
    kp_labels[kp[0]] = kp[1]["label"]
print(kp_labels)

with open('wrist_cam_1600951613.png.json') as f:
  ann = json.load(f)


# parse predictions

data_openpifpaf = [{"keypoints": [761.31, 128.4, 0.95, 767.27, 120.0, 0.87, 760.54, 120.06, 0.3, 793.71, 119.45, 0.88, 0.0, 0.0, 0.0, 800.01, 175.22, 0.82, 812.67, 162.02, 0.5, 800.48, 251.25, 0.74, 0.0, 0.0, 0.0, 754.31, 271.28, 0.4, 0.0, 0.0, 0.0, 794.72, 304.22, 0.69, 797.45, 301.43, 0.38, 769.61, 381.48, 0.74, 778.86, 372.72, 0.47, 778.82, 470.71, 0.78, 838.08, 424.9, 0.58], "bbox": [744.61, 113.27, 109.91, 374.29], "score": 0.631, "category_id": 1}, {"keypoints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 514.35, 145.7, 0.37, 605.86, 163.63, 0.45, 434.1, 246.18, 0.89, 619.17, 258.02, 0.7, 396.92, 410.43, 0.92, 753.39, 272.25, 0.6, 393.33, 493.57, 0.98, 692.87, 163.56, 0.62, 488.2, 558.21, 0.67, 607.81, 550.96, 0.58, 514.22, 702.44, 0.38, 601.03, 702.23, 0.48, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "bbox": [362.86, 130.54, 414.67, 607.17], "score": 0.575, "category_id": 1}]

x_pairs = []
y_pairs = []

for obj_i,obj in enumerate(data_openpifpaf):
  ref_openpifpaf = obj["keypoints"]
  ref_openpifpaf = list(zip(ref_openpifpaf[0::3], ref_openpifpaf[1::3]))
  print(ref_openpifpaf)

  # parse reference annotations

  points = dict()
  for id in kp_labels.values():
    points[id] = [0.0, 0.0]

  keypoints = []
  object = ann["objects"][obj_i]
  for kp in object["nodes"].items():
    keypoints.extend(kp[1]["loc"])
    points[kp_labels[kp[0]]] = kp[1]["loc"]

  points = {int(k):v for k,v in points.items()}
  print(dict(sorted(points.items())))

  res = []
  for v in points.values():
    v.append(1)
    res.extend(v)
  print(res, len(res))

  # compute pairs of point between ref and prediction

  for ref_i in mapping_ann_openpifpaf.keys():
    pred_i = mapping_ann_openpifpaf[ref_i]
    try:
      pred_x = ref_openpifpaf[pred_i][0]
      pred_y = ref_openpifpaf[pred_i][1]
      ref_x = points[ref_i][0]
      ref_y = points[ref_i][1]
      x_pairs.append((ref_x,pred_x))
      y_pairs.append((ref_y,pred_y))
    except IndexError:
      pass

  print(y_pairs)

# write to image

import cv2
image = cv2.imread('wrist_cam_1600951613.png')

for x,y in zip(x_pairs, y_pairs):
  ref = (int(x[0]),int(y[0]))
  pred = (int(x[1]),int(y[1]))
  image = cv2.circle(image, ref, radius=0, color=(0, 255, 0), thickness=10)
  image = cv2.circle(image, pred, radius=0, color=(255, 0, 0), thickness=10)

cv2.imwrite("test.png", image)
