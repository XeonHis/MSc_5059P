import numpy as np


def generate_pure_depth_image(dirpath):
    import os
    import cv2
    for file in os.listdir(dirpath):
        filepath = os.path.join(dirpath, file)
        store_path = os.path.join(dirpath, "..", "depth")
        image = cv2.imread(filepath, -1)
        cv2.imwrite(os.path.join(store_path, file), image[:, :, -1])


def pcd_visualize(filepath):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])


def read_npy(filepath):
    import numpy as np
    npy_data = np.load(filepath)
    print(npy_data)


def convert_xyzrgb_to_txt(filepath):
    import numpy as np
    data = np.loadtxt(filepath)
    data[:, 3:] = np.around(data[:, 3:] * 255)
    # print(data)
    # np.savetxt(filepath.replace("xyzrgb","txt"), data)


def temp_tool(dirpath):
    import os
    start = dirpath.index("_")
    end = dirpath.rfind("/")
    prefix = dirpath[start + 1:end] + "_"
    for file in os.listdir(dirpath):
        if file.endswith(".npy"):
            original_path = os.path.join(dirpath, file)
            after_path = os.path.join(dirpath, prefix + file)
            os.rename(original_path, after_path)


if __name__ == '__main__':
    # generate_pure_depth_image("../data/processed/realsense_tissue_roll/with_depth")
    # pcd_visualize("../data/processed/realsense_tissue_roll/pcd/frame_171.pcd")
    # read_npy(
    #     'E:\Code project\python\MSc_5059P\PointNet\data\custom\cup_frame_1165_test.npy')
    pass
