def generate_pure_depth_image(dirpath):
    import os
    import cv2
    import numpy as np
    for file in os.listdir(dirpath):
        filepath = os.path.join(dirpath, file)
        store_path = os.path.join(dirpath, "..", "depth")
        image = cv2.imread(filepath, -1)
        cv2.imwrite(os.path.join(store_path, file), image[:, :, -1])


def pcd_visualize(filepath):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    # generate_pure_depth_image("../data/processed/realsense_tissue_roll/with_depth")
    # pcd_visualize("../data/processed/realsense_tissue_roll/pcd/frame_171.pcd")
    pass
