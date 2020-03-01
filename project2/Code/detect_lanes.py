#!/usr/bin/env python3

import os
import argparse
import cv2
import numpy as np
from custom import utils, file_utils, pre_process, lines

# defaults (for testing)
CALIBRATION="../Data/data_1/camera_params.yaml"
VIDEO="../Data/data_1/data_1.mp4"
# manually selected lane points
LANE_POINTS=[[0,422],[589,256],[694,244],[840,510]]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-V","--video",type=str,default=VIDEO,help="Video file to process")
    parser.add_argument("-v","--verbosity",type=int,default=1,help="Verbosity")
    parser.add_argument("-c","--calibration",type=str,default=CALIBRATION,help="Yaml file containing camera intrinsics.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # get camera calibration
    K,D = file_utils.load_params(args.calibration)

    # initialize our video IO
    vidgen = file_utils.VidGenerator(args.video, args.verbosity)
    output_file = "processed_" + os.path.basename(args.video)
    video_writer = file_utils.VidWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), vidgen.fps, vidgen.size, isColor=True)

    # compute homography (from manual points)
    subsect = utils.get_subsection(vidgen.size,400,250)
    H,_ = cv2.findHomography(np.array(LANE_POINTS),subsect) 

    # step through each frame and process
    with video_writer as writer:
        for ret,frame in vidgen:
            # utils.plot(frame, "Frame")
            
            # undistort image
            rect = pre_process.rectify(frame, K, D)
            # utils.plot(rect, "Rect")

            # convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # utils.plot(gray, "Grayscale")

            # warp perspective (convert lanes to "vertical")
            warped = cv2.warpPerspective(gray, H, dsize=vidgen.size)
            # utils.plot(warped, "Warped")

            # denoise the image
            blur = cv2.GaussianBlur(warped,(7,7),cv2.BORDER_DEFAULT)
            # utils.plot(blur, "Blur")
            
            # threshold
            ret, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
            # utils.plot(thresh, "thresh")
           
            # edge detection
            # edges = cv2.Canny(thresh, 50, 125)
            # utils.plot(edges, "Edge Detection")
            
            # fit lines (via column histogram method)
            result,fits = lines.polyfit(thresh)
            utils.plot(result, "Detections")

            writer.write(result)

