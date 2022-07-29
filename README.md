# University Of Glasgow MSc 5059P - Object Detection and Size Estimation

### 数据集采集与构建步骤
1. 获取整个.bag文件
2. 运行```align_color2depth.py```读取.bag文件，将RGB与Depth信息进行融合展示，并通过Enter从流中截取感兴趣的图像帧
3. 运行```depth_to_point_cloud.py```将RGBD图片重构为点云
4. 使用CloudCompare进行点云标记，label为需识别的物体+bg，value均设置为1，融合后downsample导出为.txt文件，保存至与.pcd同一目录下
5. 使用```utils.tools.py#build_dataset```进行数据集构建，数据集文件为.npy，注意需根据label物品的增加增加```utils.tools.py#build_label```中的label值
