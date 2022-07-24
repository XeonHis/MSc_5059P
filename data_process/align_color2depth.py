import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os.path

# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream fps and format to match the recorded.")
# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
# Parse the command line arguments to an object
args = parser.parse_args()
# set default .bag file
args.input = "../data/raw/realsense_screwdriver.bag"

# 根据物品名称确定储存的路径
processed_data_dir = "../data/processed/"
current_store_path = processed_data_dir + args.input.split("/")[-1].split(".")[0]

# rgbd图像储存路径
with_depth_store_dir = current_store_path + "/with_depth"

# rgb图像储存路径
rgb_store_dir = current_store_path + "/rgb"

# depth图像储存路径
depth_store_dir = current_store_path + "/depth"

# 创建目录
if not os.path.exists(current_store_path):
    os.mkdir(current_store_path)
    os.mkdir(with_depth_store_dir)
    os.mkdir(rgb_store_dir)
    os.mkdir(depth_store_dir)

try:
    # Create pipeline
    pipeline = rs.pipeline()

    # Create a config object
    config = rs.config()

    # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
    rs.config.enable_device_from_file(config, args.input)

    # Start streaming from file
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Create opencv window to render image in
    cv2.namedWindow("Color Stream with Depth", cv2.WINDOW_AUTOSIZE)

    # Streaming loop
    i = 0
    while True:
        i = i + 1
        # Get frameset of depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(
            aligned_depth_frame.get_data())  # [raw_depth_info], depth = raw_depth_info * Depth_Scale
        color_image = np.asanyarray(color_frame.get_data())  # [r, g, b]

        blended_image = np.dstack((color_image, depth_image))
        blended_image_copy = blended_image.copy()

        # Render image in opencv window
        cv2.imshow("Color Stream with Depth", blended_image_copy.astype(np.uint8))
        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27:
            cv2.destroyAllWindows()
            break
        # press Enter to store image
        if key == 13:
            cv2.imwrite(with_depth_store_dir + "/frame_" + str(i) + ".png", blended_image)
            cv2.imwrite(rgb_store_dir + "/frame_" + str(i) + ".png", color_image)
            cv2.imwrite(depth_store_dir + "/frame_" + str(i) + ".png", blended_image[:, :, -1])
            print("image at frame " + str(i) + " stored.")

finally:
    pass
