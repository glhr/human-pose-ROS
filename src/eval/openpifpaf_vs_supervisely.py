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

THRESHOLD = 1/15

def parse_annotation_meta_json():
    kp_labels = dict()

    with open('meta.json') as f:
      meta = json.load(f)

    for cls in meta["classes"]:
      if not cls["title"] == "pose":
        continue
      kps = cls["geometry_config"]["nodes"]
      for kp in kps.items():
        kp_labels[kp[0]] = kp[1]["label"]
    return kp_labels

def parse_prediction_json(method, img_name, debug=False):
    with open(f"{method}/{img_name}.predictions.json") as f:
        data_predictions = json.load(f)
    predictions_list = dict()
    predictions_dict = dict()
    if debug: print(f"\n\n---------------- PREDICTIONS ----------------")
    # parse predicted keypoints
    for obj_i,obj in enumerate(data_predictions):
        predictions_dict[obj_i] = dict()
        if debug: print(f"--- Person {obj_i} ---")
        pnts_predictions = obj["keypoints"]
        pnts_predictions = list(zip(pnts_predictions[0::3], pnts_predictions[1::3]))
        if debug: print("Predicted points:",pnts_predictions, "\n")
        predictions_list[obj_i] = pnts_predictions
        for kp_id, kp in enumerate(pnts_predictions):
            predictions_dict[obj_i][kp_id] = kp
    if debug: logger.debug(predictions_dict)
    return predictions_dict, predictions_list

def parse_annotation_json(supervisely_json, kp_labels, debug=True):

    with open(supervisely_json) as f:
        ann = json.load(f)

    if debug: print(f"\n\n---------------- REFERENCE ----------------")
    ground_truths_per_instance_dict, ground_truths_dict  = dict(), dict()
    ground_truths_per_instance_list, ground_truths_list = dict(), dict()
    person_dimensions_per_instance_dict, person_dimensions_dict = dict(), dict()
    distances_per_person = dict()
    person_mappings = dict()
    pnt_pairs_matched = dict()
    ref_pnts = {'x':dict(), 'y':dict()}
    # parse reference annotations
    for obj_i,obj in enumerate(ann["objects"]):

        # create dictionary of points {id: [x,y]}
        pnts_ref = dict()
        for id in kp_labels.values():
            pnts_ref[id] = [0.0, 0.0]

        instance = obj.get("instance","")
        if not len(instance):
            continue

        if obj["classTitle"] == "pose":
            for kp in obj["nodes"].items():
                pnts_ref[kp_labels[kp[0]]] = kp[1]["loc"]

            pnts_ref = {int(k):v for k,v in pnts_ref.items()}
            pnts_ref = dict(sorted(pnts_ref.items()))
            ground_truths_per_instance_dict[instance] = pnts_ref
            ground_truths_per_instance_list[instance] = [tuple(v) for v in pnts_ref.values()]

            # add confidence
            # for v in pnts_ref.values():
            #     v.append(1)
            # print("Ref points:",pnts_ref, "\n")
        elif obj["classTitle"] == "person" and obj["geometryType"] == "rectangle":
            points = obj["points"]["exterior"]
            h = points[1][1]-points[0][1]
            w = points[1][0]-points[0][0]
            person_dimensions_per_instance_dict[instance] = {'height':h, 'width':w}

    for i, instance in enumerate(ground_truths_per_instance_dict):
        if debug: print(f"--- Person {i} ---")
        ground_truths_dict[i] = ground_truths_per_instance_dict[instance]
        ground_truths_list[i] = ground_truths_per_instance_list[instance]
        if debug: logger.debug(f"Keypoints: {ground_truths_dict[i]} ")

        person_dimensions_dict[i] = person_dimensions_per_instance_dict.get(instance,{})
        if not len(person_dimensions_dict[i]):
            person_dimensions_dict[i] = {'height':0, 'width':0}
        if debug: logger.debug(f"Dimensions: {person_dimensions_dict[i]}")



    return ground_truths_dict, ground_truths_list, person_dimensions_dict

def visualize_points(image, predictions, ground_truths, person_dimensions, img_name, debug=False):
    colors = dict()
    for k in range(18):
      colors[k] = tuple(np.random.randint(256, size=3))

    if debug: print(ground_truths)
    for person_id in ground_truths.keys():
        error_threshold = int(person_dimensions[person_id]['width']*THRESHOLD)
        for kp_id, kp in enumerate(ground_truths[person_id]):
            color = tuple(map(int, colors[kp_id]))
            ref = (int(kp[0]),int(kp[1]))
            if (ref[0] > 0 or ref[1] > 0):  # ignore (0,0) keypoints
                image = cv2.circle(image, ref, radius=error_threshold, color=(0,0,0), thickness=2)
                image = cv2.drawMarker(image, ref, color=color, thickness=2, markerSize=20)

    for person_id in predictions.keys():
        for kp_id, kp in enumerate(predictions[person_id]):
            color = tuple(map(int, colors[kp_id]))
            pred = (int(kp[0]),int(kp[1]))
            if (pred[0] > 0 or pred[1] > 0):
                image = cv2.circle(image, pred, radius=0, color=color, thickness=10)
                image = cv2.circle(image, pred, radius=0, color=(0,0,0), thickness=4)

    cv2.imwrite(f"{method}/{img_name}", image)


def reorder_keypoints_from_mappings(mapping_ann, predictions):
    predictions_sorted = dict()
    predictions_list_sorted = dict()
    for person_id, val in predictions.items():
        predictions_sorted[person_id] = dict()
        predictions_list_sorted[person_id] = [0] * len(val)
        for kp_id in val:
            predictions_sorted[person_id][kp_id] = predictions[person_id][mapping_ann[kp_id]]
            predictions_list_sorted[person_id][kp_id] = predictions[person_id][mapping_ann[kp_id]]

    logger.info(predictions_sorted)
    return predictions_sorted, predictions_list_sorted

def distance_between_keypoints(keypoints_1, keypoints_2, debug=False):
    pnt_distances = []
    for pnt_pair in zip(keypoints_1.values(), keypoints_2.values()):
        if debug: print(pnt_pair)
        pnt_distances.append(distance.euclidean(pnt_pair[0], pnt_pair[1]))
    if debug: print("--> Avg. distance", np.mean(pnt_distances))
    avg_distance = np.mean(pnt_distances)
    return avg_distance

def eval(method):

    mpjpe = []
    for file_id, supervisely_json in enumerate(glob.glob("supervisely/*.json")):
        # if file_id>1:
        #     break
        logger.info(supervisely_json)

        kp_labels= parse_annotation_meta_json()

        img_name = supervisely_json.split("/")[-1].replace('.json','')

        predictions_dict, predictions_list = parse_prediction_json(method, img_name, debug=True)
        predictions_dict, predictions_list = reorder_keypoints_from_mappings(mapping_ann[method], predictions_dict)
        ground_truths_dict, ground_truths_list, person_dimensions_dict = parse_annotation_json(supervisely_json, kp_labels)

        person_mappings = dict()
        for person_ref, keypoints_ref in ground_truths_dict.items():
            distances_to_pred = dict()
            for person_pred, keypoints_pred in predictions_dict.items():
                distances_to_pred[person_pred] = distance_between_keypoints(keypoints_ref, keypoints_pred)
                print(f"Person {person_ref} -> {person_pred}: {distances_to_pred[person_pred]}")
            person_mappings[person_ref] = min(distances_to_pred, key=distances_to_pred.get)
        logger.info(person_mappings)

        # calculate MPJPE
        person_distances = []
        for person_ref, person_pred in person_mappings.items():
            distance = distance_between_keypoints(ground_truths_dict[person_ref], predictions_dict[person_pred])
            person_distances.append(distance)
        mpjpe_per_img = np.mean(person_distances)
        logger.info(f"MPJPE: {mpjpe_per_img}")
        if not np.isnan(mpjpe_per_img):
            mpjpe.append(mpjpe_per_img)

          #
          # for person_openpifpad, pnts_openpifpaf in predictions.items():
          #   x_pairs = []
          #   y_pairs = []
          #   pnt_pairs = []
          #   # compute pairs of point between ref and prediction
          #
          #   for ref_i in mapping_ann[method].keys():
          #     pred_i = mapping_ann[method][ref_i]
          #     try:
          #       pred_x = pnts_openpifpaf[pred_i][0]
          #       pred_y = pnts_openpifpaf[pred_i][1]
          #       ref_x = pnts_ref[ref_i][0]
          #       ref_y = pnts_ref[ref_i][1]
          #       pnt_pairs.append([(ref_x, ref_y),(pred_x, pred_y)])
          #       x_pairs.append((ref_x,pred_x))
          #       y_pairs.append((ref_y,pred_y))
          #     except IndexError:
          #       pass
          #
          #   #print("--> Y pairs:", y_pairs, "\n")
          #   # print("--> X pairs:", x_pairs, "\n")
          #
          #   # print("--> Point pairs:")
          #   pnt_distances = []
          #   for pnt_pair in pnt_pairs:
          #     # print(pnt_pair)
          #     pnt_distances.append(distance.euclidean(pnt_pair[0], pnt_pair[1]))
          #   # print("--> Avg. distance", np.mean(pnt_distances))
          #   avg_distance = np.mean(pnt_distances)
          #   if not distances_per_person.get(obj_i,0) or avg_distance < distances_per_person[obj_i]:
          #     distances_per_person[obj_i] = avg_distance
          #     person_mappings[obj_i] = person_openpifpad
          #     pnt_pairs_matched[obj_i] = pnt_pairs
          #     ref_pnts['x'][obj_i] = [pair[0] for i,pair in enumerate(x_pairs) if pair[0]>0 or y_pairs[i][0]>0]
          #     ref_pnts['y'][obj_i] = [pair[0] for i,pair in enumerate(y_pairs) if pair[0]>0 or x_pairs[i][0]>0]
          #


          # find minimum average distance across detected skeletons
          # distances_per_person.append(distances)

        # print("\nAvg. MPJPE (pixels) per person:",distances_per_person)
        # print("\nPerson correspondences (ref:pred):",person_mappings)
        # print("\nPerson point pairs:",pnt_pairs_matched)
        #
        # error_radius = dict()
        #
        # for person_id, person_pairs in pnt_pairs_matched.items():
        #   print(f"--- Person {person_id} ---")
        #   #print(ref_pnts['x'][person_id])
        #   #print(ref_pnts['y'][person_id])
        #   x_span = (np.min(ref_pnts['x'][person_id]),np.max(ref_pnts['x'][person_id]))
        #   person_width = x_span[1]-x_span[0]
        #   y_span = (np.min(ref_pnts['y'][person_id]),np.max(ref_pnts['y'][person_id]))
        #   person_height = y_span[1]-y_span[0]
        #
        #   error_threshold = 20
        #   error_radius[person_id] = int((error_threshold))
        #   print(error_radius[person_id])
        #
        #   print('width:',person_width)
        #   print('height:',person_height)
        #   correct_keypoints = 0
        #   print("ground truth\tprediction\t\terror")
        #   for pnt_pair in person_pairs:
        #     error = distance.euclidean(pnt_pair[0], pnt_pair[1])
        #     print(f"{pnt_pair[0][0]:4.0f},{pnt_pair[0][1]:4.0f}\t{pnt_pair[1][0]:7.2f},{pnt_pair[1][1]:7.2f}\t\t{error:.2f}")
        #     if error < error_threshold and pnt_pair[0][0] > 0 and pnt_pair[0][1]>0:
        #       correct_keypoints += 1
        #
        #   print(f"--> {correct_keypoints} correct keypoints / {len(ref_pnts['x'][person_id])} ground truth keypoints")

        # write to image
        logger.debug(f"supervisely/{img_name}")
        image = cv2.imread(f"supervisely/{img_name}")

        visualize_points(image, predictions_list, ground_truths_list, person_dimensions_dict, img_name)

    logger.info(f"MPJPE across images for {method}: {np.mean(mpjpe)}")

eval(method)
