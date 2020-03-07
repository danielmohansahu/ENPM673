#!/usr/bin/env python3
"""Process the given data set #1
"""
import os
import glob
import argparse
from custom import file_utils, lane, process_video

# File Locations
CALIBRATION="data/data1_intrinsics.yaml"
IMAGE_DIR="../Data/data_1/data/"
VIDEOFILE="data_1.mp4"

# manually selected lane points (to bootstrap tracker)
LANE_POINTS=[[223,508],[615,258],[697,265],[920,510]]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--image_dir",type=str,default=IMAGE_DIR,help="Location of images.")
    parser.add_argument("-t","--image_type",type=str,default="png",help="Image file type.")
    parser.add_argument("-d","--debug",type=bool,default=False,help="Enable/Disable debugging.")
    parser.add_argument("-c","--calibration",type=str,default=CALIBRATION,help="Yaml file containing camera intrinsics.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # get camera calibration
    K,D = file_utils.load_params(args.calibration)

    if not os.path.isfile(VIDEOFILE):
        # convert images to a video (for faster processing)
        images_regex = args.image_dir + "*" + args.image_type
        image_files = glob.glob(images_regex)
        if len(image_files) == 0:
            raise RuntimeError("No images found matching {}".format(images_regex))
        image_files.sort() 

        # convert to video
        print("Converting images to a video...")
        file_utils.pics2video(image_files, VIDEOFILE)

    # rough slope tracking, initialized to our first frame lanes
    left = lane.Lane(LANE_POINTS[0] + LANE_POINTS[1])
    right = lane.Lane(LANE_POINTS[2] + LANE_POINTS[3])
    
    print("Processing {}".format(VIDEOFILE))
    process_video(VIDEOFILE, (K,D), left, right, debug=args.debug)
    print("Finished processing images.")

