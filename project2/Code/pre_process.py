#!/usr/bin/env python3
"""Utility function to 'pre-process' a given video.
"""

import os
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
from custom import file_utils, process, utils

DEFAULT_VIDEO="../Data/Night Drive - 2689.mp4"

def parse_args():
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--videofile", type=str, default=DEFAULT_VIDEO,help="Video to pre-process.")
    parser.add_argument("-d", "--debug", type=bool, default=False, help="Set debugging on/off.")
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    args = parse_args()

    # initialize our video IO
    vidgen = file_utils.VidGenerator(args.videofile, args.debug)
    output_file = "processed_" + os.path.basename(args.videofile)
    video_writer = file_utils.VidWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), vidgen.fps, (vidgen.size[0]*2,vidgen.size[1]), isColor=True)

    # step through each frame and process
    with video_writer as writer:
        for ret,frame in vidgen:
            if args.debug:
                utils.plot(frame, "Original Image")
            result = frame.copy()
            
            # gamma correction
            result = process.gamma_correct(result)
            if args.debug:
                utils.plot(result, "Gamma correction.")
            
            # convert to grayscale
            result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
            if args.debug:
                utils.plot(result, "Grayscale Image")
            
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16,16))
            result = clahe.apply(result)
            if args.debug:
                utils.plot(result, "Histogram Equalization")

            # blur out noise
            result = cv2.GaussianBlur(result,(7,7),cv2.BORDER_DEFAULT)
            if args.debug:
                utils.plot(result, "Blurred Image")

            # write to a new video file
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
            result = np.hstack((frame,result))
            writer.write(result)

