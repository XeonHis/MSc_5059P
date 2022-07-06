from models.pointnet2_utils import farthest_point_sample, square_distance, index_points
import torch


def test_farthest_point_sample():
    test_xyz = torch.rand(64, 1000, 3)
    test_npoint = 1000
    farthest_point_sample(test_xyz, test_npoint)


def test_square_distance():
    test_src = torch.rand(64, 500, 3)
    test_dst = torch.rand(64, 1000, 3)
    square_distance(test_src, test_dst)


def test_index_points():
    test_points = torch.rand(64, 500, 3)
    test_idx = torch.rand(64, 2, 40, 50, 60)
    index_points(test_points, test_idx)


if __name__ == '__main__':
    # test_farthest_point_sample()
    # test_square_distance()
    test_index_points()
