import argparse
import os
from data_utils.S3DISDataLoader import InferenceDataset
import torch
import sys
import importlib
from tqdm import tqdm
import numpy as np
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


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--num_votes', type=int, default=1,
                        help='aggregate segmentation scores with voting [default: 5]')

    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def main(args):
    '''HYPER PARAMETER'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    experiment_dir = 'log/sem_seg/' + args.log_dir

    NUM_CLASSES = len(classes)
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    filepath = 'data/custom/inference/magroll_frame_13610.1ds_test.npy'
    data = np.load(filepath)

    TEST_DATASET_WHOLE_SCENE = InferenceDataset(data)

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).to(device)
    checkpoint = torch.load(
        str(experiment_dir) + '/checkpoints/best_model.pth') if torch.cuda.is_available() \
        else torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', map_location=torch.device('cpu'))
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    with torch.no_grad():

        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
        total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]

        whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[0]  # [x,y,z,r,g,b]
        whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[0]  # [label]
        vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
        for _ in tqdm(range(args.num_votes), total=args.num_votes):
            scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[0]
            num_blocks = scene_data.shape[0]
            s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
            batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))

            batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
            batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
            batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

            for sbatch in range(s_batch_num):
                start_idx = sbatch * BATCH_SIZE
                end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                real_batch_size = end_idx - start_idx
                batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                batch_data[:, :, 3:6] /= 1.0

                torch_data = torch.Tensor(batch_data)
                torch_data = torch_data.float().to(device)
                torch_data = torch_data.transpose(2, 1)
                seg_pred, _ = classifier(torch_data)
                batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                           batch_pred_label[0:real_batch_size, ...],
                                           batch_smpw[0:real_batch_size, ...])

        pred_label = np.argmax(vote_label_pool, 1)

        result = np.copy(whole_scene_data)
        for i in range(whole_scene_label.shape[0]):
            color = g_label2color[pred_label[i]]
            result[i, 3:] = color[0:]
        pointcloud_visualization(result)


if __name__ == '__main__':
    args = parse_args()
    main(args)
