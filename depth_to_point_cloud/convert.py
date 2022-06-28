import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
import os

file_dir = "../data/processed/realsense_tissue_roll"
rgb_image_dir = os.path.join(file_dir, "rgb")
depth_image_dir = os.path.join(file_dir, "depth")
pcd_image_dir = os.path.join(file_dir, "pcd")
if not os.path.exists(pcd_image_dir):
    os.mkdir(pcd_image_dir)

for filename in os.listdir(rgb_image_dir):
    rgb_raw = o3d.io.read_image(os.path.join(rgb_image_dir, filename))
    depth_raw = o3d.io.read_image(os.path.join(depth_image_dir, filename))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=rgb_raw, depth=depth_raw, depth_scale=1000,
                                                                    depth_trunc=100, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # print(pcd)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # o3d.visualization.draw_geometries([pcd])

    o3d.io.write_point_cloud(os.path.join(pcd_image_dir, filename.replace("png", "pcd")), pcd)
    # o3d.io.write_point_cloud(os.path.join(pcd_image_dir, filename.replace("png", "xyzrgb")), pcd)
