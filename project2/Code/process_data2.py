#!/usr/bin/env python3
"""Process the "challenge_video" with lane detection.
"""

import argparse
from custom import file_utils, lane, process_video

# defaults (for testing)
CALIBRATION="data/data2_intrinsics.yaml"
VIDEO="../Data/data_2/challenge_video.mp4"
# manually selected lane points
LANE_POINTS=[[315,695],[603,494],[720,469],[1077,699]]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-V","--video",type=str,default=VIDEO,help="Video file to process")
    parser.add_argument("-d","--debug",type=bool,default=False,help="Enable/Disable debugging.")
    parser.add_argument("-c","--calibration",type=str,default=CALIBRATION,help="Yaml file containing camera intrinsics.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # get camera calibration
    K,D = file_utils.load_params(args.calibration)

    # rough slope tracking, initialized to our first frame lanes
    left = lane.Lane(LANE_POINTS[0] + LANE_POINTS[1])
    right = lane.Lane(LANE_POINTS[2] + LANE_POINTS[3])
    
    print("Processing {}".format(args.video))
    process_video(args.video, (K,D), left, right, debug=args.debug)

