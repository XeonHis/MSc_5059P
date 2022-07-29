import copy
import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset


class S3DISDataset(Dataset):
    def __init__(self, data_root, split='train', num_point=4096, block_size=1.0,
                 sample_rate=1.0, transform=None):
        """
        init
        :param data_root: 数据集的根目录
        :param split: train or test
        :param num_point:
        :param block_size: cube的边长，用于对空间进行分割
        :param sample_rate:
        :param transform:
        """
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        # 读取所有的npy文件
        files = sorted(os.listdir(data_root))
        # 根据文件名划分train，test
        if split == 'train':
            files_split = [file for file in files if '_test' not in file]
        else:
            files_split = [file for file in files if '_test' in file]

        self.object_points, self.object_labels = [], []
        self.object_coord_min, self.object_coord_max = [], []
        num_point_all = []
        label_weights = np.zeros(9)

        for file_name in tqdm(files_split, total=len(files_split)):
            object_path = os.path.join(data_root, file_name)
            object_data = np.load(object_path)  # [x,y,z,r,g,b,label], N*7
            points, labels = object_data[:, 0:6], object_data[:, 6]  # [x,y,z,r,g,b], N*6; [label], N
            # 直方图，统计各个类别的个数,range确定最小值和最大值（不包含）
            tmp, _ = np.histogram(labels, range(10))
            label_weights += tmp
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]  # 坐标的3个最小值和最大值
            self.object_points.append(points), self.object_labels.append(labels)
            self.object_coord_min.append(coord_min), self.object_coord_max.append(coord_max)
            num_point_all.append(labels.size)
        label_weights = label_weights.astype(np.float32)
        label_weights = label_weights / np.sum(label_weights)  # 归一化
        self.label_weights = np.power(np.amax(label_weights) / label_weights, 1 / 3.0)
        print(self.label_weights)
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        idxes = []
        for index in range(len(files_split)):
            idxes.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.idxes = np.array(idxes)
        print("Totally {} samples in {} set.".format(len(self.idxes), split))

    def __getitem__(self, idx):
        room_idx = self.idxes[idx]
        points = self.object_points[room_idx]  # N * 6
        labels = self.object_labels[room_idx]  # N
        N_points = points.shape[0]

        while True:
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            # 找到位于block内的点
            point_idxes = np.where(
                (points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (
                        points[:, 1] <= block_max[1]))[0]

            if point_idxes.size > 1024:
                break

        if point_idxes.size >= self.num_point:
            selected_point_idxes = np.random.choice(point_idxes, self.num_point, replace=False)
        else:
            selected_point_idxes = np.random.choice(point_idxes, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxes, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.object_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.object_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.object_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxes]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.idxes)


class ScannetDatasetWholeScene:
    # prepare to give prediction on each points
    def __init__(self, root, block_points=4096, split='test', stride=0.5, block_size=1.0, padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.root = root
        self.split = split
        self.stride = stride
        self.scene_points_num = []
        assert split in ['train', 'test']
        if self.split == 'train':
            self.file_list = [d for d in os.listdir(root) if d.find('_test') == -1]
        else:
            self.file_list = [d for d in os.listdir(root) if d.find('_test') != -1]
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []
        for file in self.file_list:
            data = np.load(root + file)
            points = data[:, :3]
            self.scene_points_list.append(data[:, :6])
            self.semantic_labels_list.append(data[:, 6])
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        label_weights = np.zeros(9)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(10))
            self.scene_points_num.append(seg.shape[0])
            label_weights += tmp
        label_weights = label_weights.astype(np.float32)
        label_weights = label_weights / np.sum(label_weights)
        self.label_weights = np.power(np.amax(label_weights) / label_weights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:, :6]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]), np.array([])
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxes = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (
                            points[:, 1] >= s_y - self.padding) & (
                            points[:, 1] <= e_y + self.padding))[0]
                if point_idxes.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxes.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxes.size <= point_idxes.size) else True
                point_idxes_repeat = np.random.choice(point_idxes, point_size - point_idxes.size, replace=replace)
                point_idxes = np.concatenate((point_idxes, point_idxes_repeat))
                np.random.shuffle(point_idxes)
                data_batch = points[point_idxes, :]
                normalized_xyz = np.zeros((point_size, 3))
                normalized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normalized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normalized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch[:, 3:6] /= 255.0
                data_batch = np.concatenate((data_batch, normalized_xyz), axis=1)
                label_batch = labels[point_idxes].astype(int)
                batch_weight = self.label_weights[label_batch]

                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
                label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_room.size else batch_weight
                index_room = np.hstack([index_room, point_idxes]) if index_room.size else point_idxes
        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
        label_room = label_room.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_room = index_room.reshape((-1, self.block_points))
        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.scene_points_list)


class InferenceDataset:
    # prepare to give prediction on each points
    def __init__(self, data, block_points=4096, stride=0.5, block_size=1.0, padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.stride = stride
        self.scene_points_num = []

        data = data
        # xyz坐标
        points = data[:, :3]
        # 全部的点
        self.scene_points = data[:, :6]
        self.coord_min, self.coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]  # 找出坐标的最大以及最小值

        self.label_weights = np.ones(9)

    def __getitem__(self, index):

        points = self.scene_points  # [x,y,z,r,g,b]
        '''根据stride划分出来块的x和y'''
        grid_x = int(np.ceil(float(self.coord_max[0] - self.coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(self.coord_max[1] - self.coord_min[1] - self.block_size) / self.stride) + 1)
        data_per_seg, index_per_seg = np.array([]), np.array([])
        list_data_per_seg, list_index_per_seg = [], []
        '''对每一块进行遍历'''
        for index_y in range(0, grid_y):
            for index_x in range(0, grid_x):
                '''一个矩形区域'''
                s_x = self.coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, self.coord_max[0])
                s_x = e_x - self.block_size
                s_y = self.coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, self.coord_max[1])
                s_y = e_y - self.block_size
                '''查找在当前的矩形内的点'''
                point_idxes = np.where(
                    (points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) & (
                            points[:, 1] >= s_y - self.padding) & (
                            points[:, 1] <= e_y + self.padding))[0]
                if point_idxes.size == 0:
                    continue
                num_batch = int(np.ceil(point_idxes.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                '''确认点的数目需不需要replace'''
                replace = False if (point_size - point_idxes.size <= point_idxes.size) else True
                '''填充点的数量到4096'''
                point_idxes_repeat = np.random.choice(point_idxes, point_size - point_idxes.size, replace=replace)
                point_idxes = np.concatenate((point_idxes, point_idxes_repeat))
                np.random.shuffle(point_idxes)
                # 每次batch里面的data数据, [4096, 6]
                data_batch = points[point_idxes, :]
                normalized_xyz = np.zeros((point_size, 3))
                '''做归一化'''
                normalized_xyz[:, 0] = data_batch[:, 0] / self.coord_max[0]
                normalized_xyz[:, 1] = data_batch[:, 1] / self.coord_max[1]
                normalized_xyz[:, 2] = data_batch[:, 2] / self.coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                # color数据归一化到[0,1]
                data_batch[:, 3:6] /= 255.0
                data_batch = np.concatenate((data_batch, normalized_xyz), axis=1)

                # data_per_seg = np.vstack([data_per_seg, data_batch]) if data_per_seg.size else data_batch
                list_data_per_seg += list(data_batch)
                # index_per_seg = np.hstack([index_per_seg, point_idxes]) if index_per_seg.size else point_idxes
                list_index_per_seg += list(point_idxes)
        data_per_seg = np.array(list_data_per_seg)
        index_per_seg = np.array(list_index_per_seg)
        data_per_seg = data_per_seg.reshape((-1, self.block_points, data_per_seg.shape[1]))
        index_per_seg = index_per_seg.reshape((-1, self.block_points))

        return data_per_seg, index_per_seg

    def __len__(self):
        return len(self.scene_points)
