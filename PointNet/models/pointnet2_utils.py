import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # dim1和dim2交换 [B, N, M]
    dist += torch.sum(src ** 2, -1).view(B, N, 1)  # [B, N] -> [B, N, 1]
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)  # [B, M] -> [B, 1, M]
    return dist


def index_points(points, idx):
    """
    根据索引在原点云中提取点

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(
        repeat_shape)  # [B, (len(view_shape) - 1)] -> [B, S]
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    从一个输入点云中按照所需要的点的个数npoint采样出足够多的点

    Input:
        xyz: 源点云数据 pointcloud data, [B, N, 3]
        npoint: 数目 number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  # [B, npoint]
    distance = torch.ones(B, N).to(device) * 1e10  # [B, N]
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # [B] 储存当前最远点的index
    batch_indices = torch.arange(B, dtype=torch.long).to(device)  # [0, B-1]
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1,
                                                        3)  # [B, 3] -> [B, 1, 3] 根据batch_indices, farthest, : 提取出xyz
        dist = torch.sum((xyz - centroid) ** 2, -1)  # [B, N]
        mask = dist < distance
        distance[mask] = dist[mask]  # 复制mask为true的dist值到distance
        farthest = torch.max(distance, -1)[1]  # 获得最大值的index
    return centroids  # npoint个采样点在原始点云中的索引


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    寻找球形领域中的点

    Input:
        radius: 半径 local region radius

        nsample: 最多能采样的点的数目 max sample number in local region

        xyz: 源点云 all points, [B, N, 3]

        new_xyz: 领域的中心点 query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    # [B, S, N] 中心点和所有点的欧式距离
    sqrdists = square_distance(new_xyz, xyz)
    # 如果距离大于r^2，置为N
    group_idx[sqrdists > radius ** 2] = N
    # 升序排列，取出dim=2中的前nsample个点（即论文中的k个）
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # 考虑到有可能前nsample个点中也有被赋值为N的点（即球形区域内不足nsample个点），这种点需要舍弃，直接用第一个点来代替即可
    # [B, S, k]， 实际就是把group_idx中的第一个点的值复制为了[B, S, K]的维度，便利于后面的替换
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    # 找到group_idx中值等于N的点
    mask = group_idx == N
    # 将这些点的值替换为第一个点的值
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint: Number of point for FPS
        radius: Radius of ball query
        nsample: Number of point for each ball query
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    # 从原点云中挑出最远点采样的采样点为new_xyz
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)  # [B, npoint, C]
    # idx:[B, npoint, nsample] 代表npoint个球形区域中每个区域的nsample个采样点的索引
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # [B, npoint, nsample]
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    # grouped_xyz减去采样点即中心值
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  # [B, npoint, nsample, C]

    # 如果每个点上面有新的特征的维度，则用新的特征与旧的特征拼接，否则直接返回旧的特征
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            # kernel size: [1, 1]
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N] centroid
            points: input points data, [B, D, N] all points
        Return:
            new_xyz: sampled points position data, [B, C, S] sampled centroid
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]
        # 利用1x1的2d的卷积相当于把每个group当成一个通道，共npoint个通道，对[C+D, nsample]的维度上做逐像素的卷积，结果相当于对单个C+D维度做1d的卷积
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        # 对每个group做一个max pooling得到局部的全局特征，nsample: Number of point for each ball query
        # [B, output_channel, nsample, npoint] -> [B, output_channel, npoint]
        new_points = torch.max(new_points, 2)[0]
        # [B, npoint, C] -> [B, C, npoint]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        #            前面两层的质心和前面两层的输出
        """
        Input:
            利用前一层的点对后面的点进行插值
            xyz1: input points position data, [B, C, N]  l2层输出 xyz
            xyz2: sampled input points position data, [B, C, S]  l3层输出  xyz
            points1: input points data, [B, D, N]  l2层输出  points
            points2: input points data, [B, D, S]  l3层输出  points

        Return:
            new_points: upsampled points data, [B, D', N]
        """
        "  将B C N 转换为B N C 然后利用插值将高维点云数目S 插值到低维点云数目N (N大于S)"
        "  xyz1 低维点云  数量为N   xyz2 高维点云  数量为S"
        xyz1 = xyz1.permute(0, 2, 1)  # 第一次插值时 2,3,128 ---> 2,128,3 | 第二次插值时 2,3,512--->2,512,3
        xyz2 = xyz2.permute(0, 2, 1)  # 第一次插值时2,3,1  ---> 2 ,1,3    |  第二次插值时 2,3,128--->2,128,3

        points2 = points2.permute(0, 2, 1)  # 第一次插值时2,1024,1  --->2,1,1024  最后低维信息，压缩成一个点了  这个点有1024个特征
                                            # 第二次插值 2，256，128 --->2,128,256
        B, N, C = xyz1.shape  # N = 128   低维特征的点云数  （其数量大于高维特征）
        _, S, _ = xyz2.shape  # s = 1   高维特征的点云数

        if S == 1:
            "如果最后只有一个点，就将S直复制N份后与与低维信息进行拼接"
            interpolated_points = points2.repeat(1, N, 1)  # 2,128,1024 第一次直接用拼接代替插值
        else:
            "如果不是一个点 则插值放大 128个点---->512个点"
            "此时计算出的距离是一个矩阵 512x128 也就是512个低维点与128个高维点 两两之间的距离"
            dists = square_distance(xyz1, xyz2)  # 第二次插值 先计算高维与低维的距离 2,512,128
            dists, idx = dists.sort(dim=-1)  # 2,512,128 在最后一个维度进行排序 默认进行升序排序，也就是越靠前的位置说明 xyz1离xyz2距离较近
            "找到距离最近的三个邻居，这里的idx：2,512,3的含义就是512个点与128个距离最近的前三个点的索引，" \
            "例如第一行就是：对应128个点中那三个与512中第一个点距离最近"
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3] 2,512,3 此时dist里面存放的就是 xyz1离xyz2最近的3个点的距离

            dist_recip = 1.0 / (dists + 1e-8)  # 求距离的倒数 2,512,3 对应论文中的 Wi(x)
            "对dist_recip的倒数求和 torch.sum   keepdim=True 保留求和后的维度  2,512,1"
            norm = torch.sum(dist_recip, dim=2, keepdim=True)  # 也就是将距离最近的三个邻居的加起来  此时对应论文中公式的分母部分
            weight = dist_recip / norm  # 2,512,3
            """
            这里的weight是计算权重  dist_recip中存放的是三个邻居的距离  norm中存放是距离的和  
            两者相除就是每个距离占总和的比重 也就是weight
            """
            t = index_points(points2, idx)  # 2,512,3,256
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
            """
            points2: 2,128,256 (128个点 256个特征)   idx 2,512,3 （512个点中与128个点距离最近的三个点的索引）
            index_points(points2, idx) 从高维特征（128个点）中找到对应低维特征（512个点） 对应距离最小的三个点的特征 2,512,3,256
            这个索引的含义比较重要，可以再看一下idx参数的解释，其实2,512,3,256中的512都是高维特征128个点组成的。
            例如 512中的第一个点 可能是由128中的第 1 2 3 组成的；第二个点可能是由2 3 4 三个点组成的
            -------------------------------------------
            weight: 2,512,3    weight.view(B, N, 3, 1) ---> 2,512,3,1
            a与b做*乘法，原则是如果a与b的size不同，则以某种方式将a或b进行复制，使得复制后的a和b的size相同，然后再将a和b做element-wise的乘法。
            这样做乘法就相当于 512,3,256  中的三个点的256维向量都会去乘它们的距离权重，也就是一个数去乘256维向量
            torch.sum dim=2 最后在第二个维度求和 取三个点的特征乘以权重后的和 也就完成了对特征点的上采样
            """

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points
