import numpy as np


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


def build_dataset(dirpath):
    import pandas as pd
    import os
    for file in os.listdir(dirpath):
        if file.endswith(".txt"):
            current_filepath = os.path.join(dirpath, file)
            save_filepath = os.path.join(dirpath, file.replace(".txt", ""))
            data = pd.read_csv(current_filepath, sep=' ')
            result = data.apply(build_label, axis=1)
            npy_data = np.array(result[['//X', 'Y', 'Z', 'R', 'G', 'B', 'label']])
            np.save(save_filepath, npy_data)


def build_label(series):
    labels = ['bg', 'box']
    for label in labels:
        assert label in series, label + ' not in data'
    if series['bg'] == 1:
        series['label'] = 0
    if series['box'] == 1:
        series['label'] = 1
    return series


if __name__ == '__main__':
    # generate_pure_depth_image("../data/processed/realsense_tissue_roll/with_depth")
    # pcd_visualize("../data/processed/realsense_tissue_roll/pcd/frame_171.pcd")
    read_npy(
        "E:\Code project\python\Pointnet_Pointnet2_pytorch\data\s3dis\stanford_indoor3d\Area_1_WC_1.npy")
    # build_dataset("../data/processed/realsense_box/pcd")
    pass
