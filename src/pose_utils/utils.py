#! /usr/bin/env python3

import numpy as np
import vg
from vision_utils.logger import get_logger, get_printer
logger = get_logger()

try:
    import pyrealsense2 as rs
except ImportError:
    logger.warning("Can't find pyrealsense2")

def cam_to_world(cam_point, world_to_cam):
    """Convert from camera_frame to world_frame

    Keyword arguments:
    cam_pose   -- PoseStamped from camera view
    cam_frame  -- The frame id of the camera
    """
    # cam_point = np.array([cam_pose[0], cam_pose[1], cam_pose[2]])

    obj_vector = np.concatenate((cam_point, np.ones(1))).reshape((4, 1))
    world_point = np.dot(world_to_cam, obj_vector)

    world_point = [p[0] for p in world_point]
    return world_point[0:3]

def pixel_to_camera(cameraInfo, pixel, depth):
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
    x = result[0]
    y = result[1]
    depth = result[2]
    z_squared = depth**2 - x**2 - y**2
    z = np.sqrt(z_squared)
    result[2] = z

    return result

def get_points_centroid(arr):
    length = len(arr)
    arr = np.array(arr)
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    sum_z = np.sum(arr[:, 2])
    return sum_x/length, sum_y/length, sum_z/length

def angle_from_centroid(centroid, ref_vector, normal_vector):
    v0 = np.array(ref_vector)
    vcentroid = np.array(centroid)
    angle = vg.signed_angle(v0, vcentroid, look=np.array(normal_vector))
    logger.debug("Centroid angle: {}".format(angle))
    return angle

def distance_between_points(p1,p2):
    return float(vg.euclidean_distance(np.array(p1), np.array(p2)))

def vector_from_2_points(p1,p2):
    dist_v = np.subtract(p2,p1)
    norm = np.linalg.norm(dist_v)
    direction = dist_v/norm
    return direction
