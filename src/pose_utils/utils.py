#! /usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import vg

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
    x = result[0]
    depth = result[2]
    z_squared = depth**2 - x**2
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

def angle_from_centroid(centroid):
    v0 = np.array([0,0,1])
    vcentroid = np.array(centroid)
    angle = vg.signed_angle(v0, vcentroid, look=np.array([1,1,0]))
    logger.debug(f"Centroid angle: {angle}")
    return angle
