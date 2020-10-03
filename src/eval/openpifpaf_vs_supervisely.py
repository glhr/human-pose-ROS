import os
import numpy as np
import json
import glob
from datetime import datetime
from pathlib import Path
import argparse
import sys
from scipy.spatial import distance
import cv2

from vision_utils.logger import get_logger
from eval.kp_mappings import mapping_ann
logger = get_logger()

method = "pytorch_Realtime_Multi-Person_Pose_Estimation"

def eval(method):

    for supervisely_json in glob.glob("supervisely/*.json"):
        logger.info(supervisely_json)
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
        # print(kp_labels)

        with open(supervisely_json) as f:
          ann = json.load(f)

        img_name = supervisely_json.split("/")[-1].replace('.json','')

        # parse predictions
        with open(f"{method}/{img_name}.predictions.json") as f:
          data_openpifpaf = json.load(f)
        #data_openpifpaf = [{"keypoints": [761.31, 128.4, 0.95, 767.27, 120.0, 0.87, 760.54, 120.06, 0.3, 793.71, 119.45, 0.88, 0.0, 0.0, 0.0, 800.01, 175.22, 0.82, 812.67, 162.02, 0.5, 800.48, 251.25, 0.74, 0.0, 0.0, 0.0, 754.31, 271.28, 0.4, 0.0, 0.0, 0.0, 794.72, 304.22, 0.69, 797.45, 301.43, 0.38, 769.61, 381.48, 0.74, 778.86, 372.72, 0.47, 778.82, 470.71, 0.78, 838.08, 424.9, 0.58], "bbox": [744.61, 113.27, 109.91, 374.29], "score": 0.631, "category_id": 1}, {"keypoints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 514.35, 145.7, 0.37, 605.86, 163.63, 0.45, 434.1, 246.18, 0.89, 619.17, 258.02, 0.7, 396.92, 410.43, 0.92, 753.39, 272.25, 0.6, 393.33, 493.57, 0.98, 692.87, 163.56, 0.62, 488.2, 558.21, 0.67, 607.81, 550.96, 0.58, 514.22, 702.44, 0.38, 601.03, 702.23, 0.48, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "bbox": [362.86, 130.54, 414.67, 607.17], "score": 0.575, "category_id": 1}]

        predictions = dict()
        print(f"\n\n---------------- PREDICTIONS ----------------")
        # parse predicted keypoints
        for obj_i,obj in enumerate(data_openpifpaf):
          print(f"--- Person {obj_i} ---")
          pnts_openpifpaf = obj["keypoints"]
          pnts_openpifpaf = list(zip(pnts_openpifpaf[0::3], pnts_openpifpaf[1::3]))
          print("Predicted points:",pnts_openpifpaf, "\n")
          predictions[obj_i] = pnts_openpifpaf

        print(f"\n\n---------------- REFERENCE ----------------")
        ground_truths = dict()
        distances_per_person = dict()
        person_mappings = dict()
        pnt_pairs_matched = dict()
        ref_pnts = {'x':dict(), 'y':dict()}
        # parse reference annotations
        for obj_i,obj in enumerate(ann["objects"]):

          print(f"--- Person {obj_i} ---")
          # create dictionary of points {id: [x,y]}
          pnts_ref = dict()
          for id in kp_labels.values():
            pnts_ref[id] = [0.0, 0.0]

          if not obj["classTitle"] == "pose":
            continue
          object = ann["objects"][obj_i]

          for kp in object["nodes"].items():
            pnts_ref[kp_labels[kp[0]]] = kp[1]["loc"]

          pnts_ref = {int(k):v for k,v in pnts_ref.items()}
          pnts_ref = dict(sorted(pnts_ref.items()))
          ground_truths[obj_i] = pnts_ref

          # add confidence
          for v in pnts_ref.values():
            v.append(1)
          print("Ref points:",pnts_ref, "\n")

          for person_openpifpad, pnts_openpifpaf in predictions.items():
            x_pairs = []
            y_pairs = []
            pnt_pairs = []
            # compute pairs of point between ref and prediction

            for ref_i in mapping_ann[method].keys():
              pred_i = mapping_ann[method][ref_i]
              try:
                pred_x = pnts_openpifpaf[pred_i][0]
                pred_y = pnts_openpifpaf[pred_i][1]
                ref_x = pnts_ref[ref_i][0]
                ref_y = pnts_ref[ref_i][1]
                pnt_pairs.append([(ref_x, ref_y),(pred_x, pred_y)])
                x_pairs.append((ref_x,pred_x))
                y_pairs.append((ref_y,pred_y))
              except IndexError:
                pass

            #print("--> Y pairs:", y_pairs, "\n")
            # print("--> X pairs:", x_pairs, "\n")

            # print("--> Point pairs:")
            pnt_distances = []
            for pnt_pair in pnt_pairs:
              # print(pnt_pair)
              pnt_distances.append(distance.euclidean(pnt_pair[0], pnt_pair[1]))
            # print("--> Avg. distance", np.mean(pnt_distances))
            avg_distance = np.mean(pnt_distances)
            if not distances_per_person.get(obj_i,0) or avg_distance < distances_per_person[obj_i]:
              distances_per_person[obj_i] = avg_distance
              person_mappings[obj_i] = person_openpifpad
              pnt_pairs_matched[obj_i] = pnt_pairs
              ref_pnts['x'][obj_i] = [pair[0] for i,pair in enumerate(x_pairs) if pair[0]>0 or y_pairs[i][0]>0]
              ref_pnts['y'][obj_i] = [pair[0] for i,pair in enumerate(y_pairs) if pair[0]>0 or x_pairs[i][0]>0]



          # find minimum average distance across detected skeletons
          # distances_per_person.append(distances)

        print("\nAvg. MPJPE (pixels) per person:",distances_per_person)
        print("\nPerson correspondences (ref:pred):",person_mappings)
        print("\nPerson point pairs:",pnt_pairs_matched)

        error_radius = dict()
        THRESHOLD = 1/15

        for person_id, person_pairs in pnt_pairs_matched.items():
          print(f"--- Person {person_id} ---")
          #print(ref_pnts['x'][person_id])
          #print(ref_pnts['y'][person_id])
          x_span = (np.min(ref_pnts['x'][person_id]),np.max(ref_pnts['x'][person_id]))
          person_width = x_span[1]-x_span[0]
          y_span = (np.min(ref_pnts['y'][person_id]),np.max(ref_pnts['y'][person_id]))
          person_height = y_span[1]-y_span[0]

          error_threshold = person_height*THRESHOLD
          error_radius[person_id] = int((error_threshold))
          print(error_radius[person_id])

          print('width:',person_width)
          print('height:',person_height)
          correct_keypoints = 0
          print("ground truth\tprediction\t\terror")
          for pnt_pair in person_pairs:
            error = distance.euclidean(pnt_pair[0], pnt_pair[1])
            print(f"{pnt_pair[0][0]:4.0f},{pnt_pair[0][1]:4.0f}\t{pnt_pair[1][0]:7.2f},{pnt_pair[1][1]:7.2f}\t\t{error:.2f}")
            if error < error_threshold and pnt_pair[0][0] > 0 and pnt_pair[0][1]>0:
              correct_keypoints += 1

          print(f"--> {correct_keypoints} correct keypoints / {len(ref_pnts['x'][person_id])} ground truth keypoints")

        # write to image
        logger.debug(f"supervisely/{img_name}")
        image = cv2.imread(f"supervisely/{img_name}")

        colors = dict()
        for k in range(18):
          colors[k] = tuple(np.random.randint(256, size=3))

        for id,pairs in pnt_pairs_matched.items():
            for k,pair in enumerate(pairs):
                color = tuple(map(int, colors[k]))
                ref = (int(pair[0][0]),int(pair[0][1]))
                pred = (int(pair[1][0]),int(pair[1][1]))
                if (ref[0] > 0 or ref[1] > 0):  # ignore (0,0) keypoints
                    image = cv2.circle(image, ref, radius=error_radius[id], color=(0,0,0), thickness=2)
                    image = cv2.drawMarker(image, ref, color=color, thickness=2, markerSize=error_radius[id])
                    # image = cv2.circle(image, ref, radius=0, color=(0,0,0), thickness=4)
                if (pred[0] > 0 or pred[1] > 0):
                    image = cv2.circle(image, pred, radius=0, color=color, thickness=10)
                    image = cv2.circle(image, pred, radius=0, color=(0,0,0), thickness=4)

        cv2.imwrite(f"{method}/{img_name}", image)

eval(method)
