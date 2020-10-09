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

method = "fastpose-zexinchen"

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

def parse_prediction_json(method, img_name, debug=False, scale=1):
    f_name = f"{method}/{img_name}.predictions.json"
    with open(f_name) as f:
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
        pnts_predictions = [(p[0]/scale,p[1]/scale) for p in pnts_predictions]

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
        predictions_list_sorted[person_id] = [0] * len(mapping_ann)
        for kp_id in mapping_ann:
            kp_pred = predictions[person_id].get(mapping_ann[kp_id],0)
            if not kp_pred:
                predictions_sorted[person_id][kp_id] = (-1, -1)
                predictions_list_sorted[person_id][kp_id] = (-1, -1)
            else:
                predictions_sorted[person_id][kp_id] = kp_pred
                predictions_list_sorted[person_id][kp_id] = kp_pred

    logger.info(predictions_sorted)
    return predictions_sorted, predictions_list_sorted

def distance_between_skeletons(keypoints_1, keypoints_2, debug=False):
    pnt_distances = []
    for pnt_pair in zip(keypoints_1.values(), keypoints_2.values()):
        if debug: print(pnt_pair)
        if not (-1 in pnt_pair[0] or -1 in pnt_pair[1]):
            pnt_distances.append(distance.euclidean(pnt_pair[0], pnt_pair[1]))
    if debug: print("--> Avg. distance", np.mean(pnt_distances))
    avg_distance = np.mean(pnt_distances)
    return avg_distance

def nck_between_skeletons(predictions, ground_truths, person_dimensions):
    error_threshold = int(person_dimensions['width']*THRESHOLD)
    correct_keypoints = 0
    logger.debug(f"Error threshold: {error_threshold}")
    for pnt_pair in zip(predictions.values(), ground_truths.values()):
        error = distance.euclidean(pnt_pair[0], pnt_pair[1])
        if error < error_threshold:
            correct_keypoints += 1
            print(f"* {pnt_pair[0][0]:4.0f},{pnt_pair[0][1]:4.0f}\t{pnt_pair[1][0]:4.0f},{pnt_pair[1][1]:4.0f}\t\t{error:.2f}")
        else:
            print(f"  {pnt_pair[0][0]:4.0f},{pnt_pair[0][1]:4.0f}\t{pnt_pair[1][0]:4.0f},{pnt_pair[1][1]:4.0f}\t\t{error:.2f}")

    total_keypoints = sum(-1.0 not in value for value in predictions.values())
    print(f"--> {correct_keypoints} correct keypoints / {total_keypoints} ground truth keypoints ({len(predictions.values())-total_keypoints} ignored)")
    return correct_keypoints, total_keypoints

def eval(method, scale=1):

    mpjpe_overall, nck_overall, totalk_overall = dict(), dict(), dict()
    mpjpe_overall['gt'], mpjpe_overall['pred']  = [], []
    nck_overall['gt'], nck_overall['pred'] = 0, 0
    totalk_overall['gt'], totalk_overall['pred'] = 0, 0
    for file_id, supervisely_json in enumerate(glob.glob("supervisely/*.json")):
        # if file_id>1:
        #     break
        logger.info(supervisely_json)

        kp_labels= parse_annotation_meta_json()

        img_name_orig = supervisely_json.split("/")[-1].replace('.json','')
        if scale == 1:
            img_name = img_name_orig
        else:
            img_name = supervisely_json.split("/")[-1].replace('.json','').split('.')[0]
            img_name = f"{img_name}-{scale}.png"

        predictions_dict, predictions_list = parse_prediction_json(method, img_name, debug=True, scale=scale)
        predictions_dict, predictions_list = reorder_keypoints_from_mappings(mapping_ann[method], predictions_dict)
        ground_truths_dict, ground_truths_list, person_dimensions_dict = parse_annotation_json(supervisely_json, kp_labels)

        ## OUT OF GROUND TRUTH

        person_mappings = dict()
        for person_ref, keypoints_ref in ground_truths_dict.items():
            distances_to_pred = dict()
            for person_pred, keypoints_pred in predictions_dict.items():
                distances_to_pred[person_pred] = distance_between_skeletons(keypoints_ref, keypoints_pred)
                print(f"Person {person_ref} -> {person_pred}: {distances_to_pred[person_pred]}")
            if len(distances_to_pred):
                person_mappings[person_ref] = min(distances_to_pred, key=distances_to_pred.get)
        logger.info(person_mappings)

        # calculate MPJPE
        person_distances = []
        for person_ref, person_pred in person_mappings.items():
            distance = distance_between_skeletons(ground_truths_dict[person_ref], predictions_dict[person_pred])
            person_distances.append(distance)
        mpjpe_per_img = np.mean(person_distances)
        logger.info(f"MPJPE: {mpjpe_per_img}")
        if not np.isnan(mpjpe_per_img):
            mpjpe_overall['gt'].append(mpjpe_per_img)

        # calculate PCK
        correct_keypoints_per_img = 0
        total_keypoints_per_img = 0
        for person_ref, person_pred in person_mappings.items():
            correct_keypoints, total_keypoints = nck_between_skeletons(predictions_dict[person_pred], ground_truths_dict[person_ref], person_dimensions_dict[person_ref])
            correct_keypoints_per_img += correct_keypoints
            total_keypoints_per_img += total_keypoints
        logger.info(f"NCK: {correct_keypoints_per_img}/{total_keypoints_per_img}")
        nck_overall['gt'] += correct_keypoints_per_img
        totalk_overall['gt'] += total_keypoints_per_img


        ## OUT OF PREDICTIONS

        person_mappings = dict()
        for person_ref, keypoints_ref in predictions_dict.items():
            distances_to_pred = dict()
            for person_pred, keypoints_pred in ground_truths_dict.items():
                distances_to_pred[person_pred] = distance_between_skeletons(keypoints_ref, keypoints_pred)
                print(f"Person {person_ref} -> {person_pred}: {distances_to_pred[person_pred]}")
            if len(distances_to_pred):
                person_mappings[person_ref] = min(distances_to_pred, key=distances_to_pred.get)
        logger.info(person_mappings)

        if len(person_mappings):
            # calculate MPJPE
            person_distances = []
            for person_pred, person_ref in person_mappings.items():
                distance = distance_between_skeletons(ground_truths_dict[person_ref], predictions_dict[person_pred])
                person_distances.append(distance)
            mpjpe_per_img = np.mean(person_distances)
            logger.info(f"MPJPE: {mpjpe_per_img}")
            if not np.isnan(mpjpe_per_img):
                mpjpe_overall['pred'].append(mpjpe_per_img)

            # calculate PCK
            correct_keypoints_per_img = 0
            total_keypoints_per_img = 0
            for person_pred, person_ref in person_mappings.items():
                correct_keypoints, total_keypoints = nck_between_skeletons(predictions_dict[person_pred], ground_truths_dict[person_ref], person_dimensions_dict[person_ref])
                correct_keypoints_per_img += correct_keypoints
                total_keypoints_per_img += total_keypoints
            logger.info(f"NCK: {correct_keypoints_per_img}/{total_keypoints_per_img}")
            nck_overall['pred'] += correct_keypoints_per_img
            totalk_overall['pred'] += total_keypoints_per_img



        # write to image
        logger.debug(f"supervisely/{img_name}")
        image = cv2.imread(f"supervisely/{img_name_orig}")

        visualize_points(image, predictions_list, ground_truths_list, person_dimensions_dict, img_name)

    logger.info(f"-- Out of ground truth --")
    logger.info(f"MPJPE across images for {method}: {np.mean(mpjpe_overall['gt'])} ({len(mpjpe_overall['gt'])} images)")
    logger.info(f"NCK across images for {method}: {nck_overall['gt']/totalk_overall['gt']:.2f}")

    logger.info(f"-- Out of predictions --")
    logger.info(f"MPJPE across images for {method}: {np.mean(mpjpe_overall['pred'])} ({len(mpjpe_overall['pred'])} images)")
    logger.info(f"NCK across images for {method}: {nck_overall['pred']/totalk_overall['pred']:.2f}")

eval(method, scale=1)
