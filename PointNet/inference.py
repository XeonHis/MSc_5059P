import os
from PointNet.data_utils.S3DISDataLoader import InferenceDataset
import torch
import sys
import importlib
import numpy as np
from datetime import datetime

from utils.tools import pointcloud_visualization

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['bg', 'box', 'magroll', 'cup', 'tissue_roll', 'umbrella', 'button', 'cupwithhandle', 'screwdriver']
g_class2color = {'box': [0 / 255, 255 / 255, 0 / 255],
                 'magroll': [0 / 255, 0 / 255, 255 / 255],
                 'cup': [0 / 255, 255 / 255, 255 / 255],
                 'tissue_roll': [255 / 255, 255 / 255, 0 / 255],
                 'umbrella': [255 / 255, 0 / 255, 255 / 255],
                 'button': [100 / 255, 100 / 255, 255 / 255],
                 'cupwithhandle': [200 / 255, 200 / 255, 100 / 255],
                 'screwdriver': [170 / 255, 120 / 255, 200 / 255],
                 'bg': [50 / 255, 50 / 255, 50 / 255]}
class2label = {cls: i for i, cls in enumerate(classes)}
g_label2color = {classes.index(cls): g_class2color[cls] for cls in classes}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def add_vote(vote_label_pool, point_idx, pred_label):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def main(data):
    from pyinstrument import Profiler

    profiler = Profiler()
    profiler.start()

    NUM_CLASSES = len(classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Inference works on ", device)

    '''HYPER PARAMETER'''
    log_dir = "2022-07-24_21-46"
    experiment_dir = 'PointNet/' + 'log/sem_seg/' + log_dir
    BATCH_SIZE = 32
    NUM_POINT = 4096
    num_votes = 1
    block_size = 0.2
    stride = block_size / 2
    print('Hyper Param:\tBATCH_SIZE: {}\tNUM_POINT: {}\tblock_size: {}\tstride: {}\t'.format(BATCH_SIZE, NUM_POINT,
                                                                                             block_size, stride))

    '''数据预处理'''
    TEST_DATASET_WHOLE_SCENE = InferenceDataset(data, block_size=block_size, stride=stride)

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)
    classifier = model.get_model(NUM_CLASSES).to(device)
    checkpoint = torch.load(
        str(experiment_dir) + '/checkpoints/best_model.pth') if torch.cuda.is_available() \
        else torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', map_location=torch.device('cpu'))
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    with torch.no_grad():

        whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points  # [x,y,z,r,g,b]
        vote_label_pool = np.zeros((whole_scene_data.shape[0], NUM_CLASSES))
        for _ in range(num_votes):
            scene_data, scene_point_index = TEST_DATASET_WHOLE_SCENE[0]  # [143,4096,9],[143,4096]
            num_blocks = scene_data.shape[0]
            s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
            batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))

            batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))

            for sbatch in range(s_batch_num):
                start_idx = sbatch * BATCH_SIZE
                end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                real_batch_size = end_idx - start_idx
                batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                batch_data[:, :, 3:6] /= 1.0

                torch_data = torch.Tensor(batch_data)
                torch_data = torch_data.float().to(device)
                torch_data = torch_data.transpose(2, 1)

                eval_start = datetime.now()
                seg_pred, _ = classifier(torch_data)
                # print('eval time consuming: %f' % (datetime.now() - eval_start).seconds)
                batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                           batch_pred_label[0:real_batch_size, ...])

        pred_label = np.argmax(vote_label_pool, 1)
        pred_prob_map = [len(np.where(pred_label == i)[0]) / len(pred_label) for i in range(NUM_CLASSES)]
        print(pred_prob_map)

        result = np.copy(whole_scene_data)
        for i in range(whole_scene_data.shape[0]):
            color = g_label2color[pred_label[i]]
            result[i, 3:] = color[0:]
        output = np.column_stack((result, pred_label))
        profiler.stop()
        profiler.print()
        # np.savetxt("output.txt", output)
        pointcloud_visualization(output, pred_prob_map)
        return output, pred_prob_map


if __name__ == '__main__':
    filepath = 'PointNet/data/custom/inference/magroll_frame_13610.02ds.npy'
    # filepath = 'PointNet/data/custom/cup_frame_856_test.npy'
    # filepath = 'PointNet/data/custom/tissue_roll_frame_739_test.npy'
    data = np.load(filepath)
    main(data)
