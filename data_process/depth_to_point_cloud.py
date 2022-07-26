import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
import os

file_dir = "../raw_data/processed/realsense_screwdriver"
rgb_image_dir = os.path.join(file_dir, "rgb")
depth_image_dir = os.path.join(file_dir, "depth")
pcd_image_dir = os.path.join(file_dir, "pcd")
if not os.path.exists(pcd_image_dir):
    os.mkdir(pcd_image_dir)

for filename in os.listdir(rgb_image_dir):
    # 读取rgb图像
    rgb_raw = o3d.io.read_image(os.path.join(rgb_image_dir, filename))
    # 读取深度图像
    depth_raw = o3d.io.read_image(os.path.join(depth_image_dir, filename))
    # 转换为rgbd图像，放大倍数1000，相机内参depth scale=0.0002500000118743628
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=rgb_raw, depth=depth_raw, depth_scale=1000,
                                                                    depth_trunc=100, convert_rgb_to_intensity=False)
    # 创建点云文件
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # print(pcd)
    # 旋转点云
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # 可视化
    # o3d.visualization.draw_geometries([pcd])
    # 输出为pcd文件
    o3d.io.write_point_cloud(os.path.join(pcd_image_dir, filename.replace("png", "pcd")), pcd)
