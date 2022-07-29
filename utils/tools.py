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
    # print(rawdata)
    # np.savetxt(filepath.replace("xyzrgb","txt"), rawdata)


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


def pointcloud_visualization(all_data, iou_map):
    import open3d as o3d

    '''从文件中读取'''
    # # 创建点云文件
    # pcd = o3d.io.read_point_cloud(data, format='xyzrgb')
    # # 边界框
    # aabb = pcd.get_axis_aligned_bounding_box()
    # aabb.color = (1, 0, 0)
    # # 可视化
    # o3d.visualization.draw_geometries([pcd,aabb])
    '''从ndarray中读取'''
    whole_scene = o3d.geometry.PointCloud()
    whole_scene.points = o3d.utility.Vector3dVector(all_data[:, :3])
    whole_scene.colors = o3d.utility.Vector3dVector(all_data[:, 3:6])
    seg_data = get_seg_data(all_data, iou_map)
    seg_scene = o3d.geometry.PointCloud()
    seg_scene.points = o3d.utility.Vector3dVector(seg_data[:, :3])
    seg_scene.colors = o3d.utility.Vector3dVector(seg_data[:, 3:6])
    bbox = seg_scene.get_axis_aligned_bounding_box()
    bbox.color = (1, 0, 0)

    l, w, h = bbox.get_extent() * 25
    print(l, w, h)

    o3d.visualization.draw_geometries([whole_scene, bbox])
    return whole_scene, bbox


def pointcloud_rt_visualization(filepath):
    import pyrealsense2 as rs
    import numpy as np
    import cv2
    from open3d import visualization, geometry, camera, io
    # todo: 加上inference部分
    from PointNet.inference import main


    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    vis = visualization.Visualizer()
    vis.create_window('PCD', width=960, height=540)
    pointcloud = geometry.PointCloud()
    geom_added = False

    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow('bgr', color_image)


        img_depth = geometry.Image(depth_image)
        img_color = geometry.Image(color_image)
        # 生成rgbd图片
        rgbd = geometry.RGBDImage.create_from_color_and_depth(color=img_color, depth=img_depth,
                                                              depth_trunc=100, convert_rgb_to_intensity=False)

        pcd = geometry.PointCloud.create_from_rgbd_image(rgbd, camera.PinholeCameraIntrinsic(
            camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        # pcd = io.read_point_cloud(filepath, format='pcd')
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pointcloud.points = pcd.points
        pointcloud.colors = pcd.colors

        pointcloud = pointcloud.random_down_sample(0.02)

        A = np.asarray(pointcloud.points)
        B = np.asarray(pointcloud.colors)

        pcd_ndarray = np.column_stack((A, B))

        data, iou_map = main(pcd_ndarray)
        whole_scene, bbox = pointcloud_visualization(data, iou_map)


        if not geom_added:
            vis.add_geometry(whole_scene)
            vis.add_geometry(bbox)
            geom_added = True

        vis.update_geometry(whole_scene)
        vis.update_geometry(bbox)
        vis.poll_events()
        vis.update_renderer()

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    pipeline.stop()
    cv2.destroyAllWindows()
    vis.destroy_window()
    del vis


def test():
    import numpy as np
    data = np.loadtxt("PointNet/log/inference_pred.txt")
    pointcloud_visualization(data)


def calculate_lwh(data, iou_map):
    import numpy as np

    # iou_map = [0.8429003, 0., 0.27847049, 0., 0., 0., 0., 0., 0.]
    cls = iou_map.index(max(iou_map[1:]))
    # data = np.load("test/result.npy")
    iou_data = np.where(data[:, -1] == cls)[0]
    x_max_idx, y_max_idx, z_max_idx = np.argmax(data[iou_data][:, :3], axis=0)
    x_min_idx, y_min_idx, z_min_idx = np.argmin(data[iou_data][:, :3], axis=0)
    # print(data)
    spice = data[iou_data][:, :-1]
    length = spice[x_max_idx][0] - spice[x_min_idx][0]
    width = spice[y_max_idx][1] - spice[y_min_idx][1]
    height = spice[z_max_idx][2] - spice[z_min_idx][2]
    print(length / 4, width / 4, height / 4)
    return spice


def get_seg_data(data, iou_map):
    import numpy as np

    cls = iou_map.index(max(iou_map[1:]))
    iou_data = np.where(data[:, -1] == cls)[0]
    spice = data[iou_data][:, :-1]
    return spice


if __name__ == '__main__':
    # pointcloud_visualization("PointNet/log/inference_pred.txt")
    pointcloud_rt_visualization(None)
    # test()
    pass
