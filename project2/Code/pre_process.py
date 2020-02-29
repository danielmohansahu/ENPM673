#!/usr/bin/env python3
"""Utility function to 'pre-process' a given video.
"""

import os
import time
import argparse
import code
import cv2
from custom import file_utils, pre_process, utils

DEFAULT_VIDEO="../Data/Night Drive - 2689.mp4"

def parse_args():
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", 
                        "--file", 
                        type=str, 
                        default=DEFAULT_VIDEO,
                        help="Video to pre-process.")
    parser.add_argument("-v", 
                        "--verbosity", 
                        type=int, 
                        default=1, 
                        help="Set verbosity level (0 is none, 1 is console output, 2 is images).")
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    args = parse_args()

    # initialize our video IO
    vidgen = file_utils.VidGenerator(args.file, args.verbosity)
    output_file = "processed_" + os.path.basename(args.file)
    video_writer = file_utils.VidWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), vidgen.fps, vidgen.size, isColor=False)

    # step through each frame and process
    with video_writer as writer:
        for ret,frame in vidgen:

            # convert to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # sharpen image
            frame = pre_process.sharpen(frame)

            # perform histogram equalization
            frame = pre_process.equalize(frame)

            # write to a new video file
            writer.write(frame)

    # code.interact(local=locals())
