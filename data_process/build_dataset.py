import pandas as pd
import os
from pandarallel import pandarallel
import numpy as np
import random


def build_dataset(dirpath, test_percent):
    pandarallel.initialize(progress_bar=False)

    start = dirpath.index("_")
    end = dirpath.rfind("/")
    prefix = dirpath[start + 1:end] + "_"
    for file in os.listdir(dirpath):
        if file.endswith(".txt"):
            current_filepath = os.path.join(dirpath, file)
            save_filepath = os.path.join(dirpath, prefix + file.replace(".txt", ""))
            data = pd.read_csv(current_filepath, sep=' ')
            result = data.parallel_apply(build_label, column_names=data.columns.values, axis=1)
            npy_data = np.array(result[['//X', 'Y', 'Z', 'R', 'G', 'B', 'label']])
            np.save(save_filepath, npy_data)
            print("####File ", save_filepath, " saved####")
    split_train_test(dirpath, test_percent)


def build_label(series, column_names):
    label_map = {'bg': 0, 'box': 1, 'magroll': 2, 'cup': 3, 'tissue_roll': 4, 'umbrella': 5, 'button': 6,
                 'cupwithhandle': 7, 'screwdriver': 8}
    labels = column_names[-2:]
    for label in labels:
        if series[label] == 1:
            series['label'] = label_map.get(label)
    return series


def split_train_test(dirpath, test_percent):
    file_list = []
    for file in os.listdir(dirpath):
        if file.endswith(".npy"):
            file_list.append(os.path.join(dirpath, file))
    for i in range(int(len(file_list) * test_percent)):
        i = random.randint(0, len(file_list) - 1)
        temp = file_list[i].replace(".npy", "_test.npy")
        os.rename(file_list[i], temp)


if __name__ == '__main__':
    # working dir: current file dir
    build_dataset("../raw_data/processed/realsense_cupwithhandle/pcd", 0.3)
