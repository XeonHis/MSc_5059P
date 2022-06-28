import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

# read image as unchanged
# image = cv2.imread("../data/processed/realsense_tissue_roll/with_depth/frame_171.png", -1)
# cv2.imwrite("color.png",image[:,:,:-1].astype(np.uint8))
# cv2.imwrite("depth.png",image[:,:,-1])
# print(image[:, :, -1])
# cv2.imshow("title", image)
# cv2.waitKey()

color_raw = o3d.io.read_image("../data/test_image/color.png")
depth_raw = o3d.io.read_image("../data/test_image/depth.png")
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color=color_raw, depth=depth_raw, depth_scale=1000,
                                                                depth_trunc=100, convert_rgb_to_intensity=True)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
print(pcd)
o3d.visualization.draw_geometries([pcd])
