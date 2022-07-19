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


def build_dataset(dirpath, test_percent):
    import pandas as pd
    import os
    start = dirpath.index("_")
    end = dirpath.rfind("/")
    prefix = dirpath[start + 1:end] + "_"
    for file in os.listdir(dirpath):
        if file.endswith(".txt"):
            current_filepath = os.path.join(dirpath, file)
            save_filepath = os.path.join(dirpath, prefix + file.replace(".txt", ""))
            data = pd.read_csv(current_filepath, sep=' ')
            result = data.apply(build_label, column_names=data.columns.values, axis=1)
            npy_data = np.array(result[['//X', 'Y', 'Z', 'R', 'G', 'B', 'label']])
            np.save(save_filepath, npy_data)
            print("####File ", save_filepath, " saved####")
    split_train_test(dirpath, test_percent)


def build_label(series, column_names):
    label_map = {'bg': 0, 'box': 1, 'magroll': 2, 'cup': 3}
    labels = column_names[-2:]
    # for label in labels:
    #     assert label in series, label + ' not in data'
    for label in labels:
        if series[label] == 1:
            series['label'] = label_map.get(label)
    return series


def split_train_test(dirpath, test_percent):
    import os
    import random
    file_list = []
    for file in os.listdir(dirpath):
        if file.endswith(".npy"):
            file_list.append(os.path.join(dirpath, file))
    for i in range(int(len(file_list) * test_percent)):
        i = random.randint(0, len(file_list) - 1)
        temp = file_list[i].replace(".npy", "_test.npy")
        os.rename(file_list[i], temp)


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
    #     'E:\Code project\python\MSc_5059P\PointNet\data\custom\magroll_frame_880.npy')
    # build_dataset("../data/processed/realsense_magroll/pcd", 0.3)
    # split_train_test("../data/processed/realsense_magroll/pcd", 0.3)
    pass
