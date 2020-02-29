#!/usr/bin/env python3

import os
import argparse
import yaml
import cv2
import numpy as np
from custom import utils, file_utils, pre_process

# defaults (for testing)
CALIBRATION="../Data/data_1/camera_params.yaml"
VIDEO="../Data/data_1/data_1.mp4"
LANE_POINTS=[[0,422],[589,256],[694,244],[840,510]]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-V","--video",type=str,default=VIDEO,help="Video file to process")
    parser.add_argument("-v","--verbosity",type=int,default=1,help="Verbosity")
    parser.add_argument("-c","--calibration",type=str,default=CALIBRATION,help="Yaml file containing camera intrinsics.")
    return parser.parse_args()

def load_params(filename):
    """Load camera parameters from a given file."""
    with open(filename, 'r') as yamlfile:
        data = yaml.safe_load(yamlfile)
    K = np.array([float(val) for val in data["K"].split()])
    K.resize([3,3])
    D = np.array([float(val) for val in data["D"].split()])
    return K,D

if __name__ == "__main__":
    args = parse_args()

    # get camera calibration
    K,D = load_params(args.calibration)

    # compute homography (from manual points)
    H,_ = cv2.findHomography(np.array(LANE_POINTS),np.array([[0,512],[0,0],[1392,0],[1392,512]])) 
    H2,_ = cv2.findHomography(np.array([[0,512],[0,0],[1392,0],[1392,512]]),np.array(LANE_POINTS)) 

    # initialize our video IO
    vidgen = file_utils.VidGenerator(args.video, args.verbosity)
    output_file = "processed_" + os.path.basename(args.video)
    video_writer = file_utils.VidWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), vidgen.fps, vidgen.size)

    # step through each frame and process
    with video_writer as writer:
        for ret,frame in vidgen:
            
            # undistort image
            frame = pre_process.rectify(frame, K, D)
            utils.plot(frame, "Frame")

            # convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # utils.plot(gray, "Grayscale")

            # denoise the image
            blur = cv2.GaussianBlur(gray,(7,7),cv2.BORDER_DEFAULT)
            # utils.plot(blur, "Blur")

            # edge detection
            edges = cv2.Canny(blur, 100, 225)
            utils.plot(edges, "Edge Detection")

            # hough line detection
            minLineLength = 100
            maxLineGap = 50
            framelines = gray.copy()
            lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
            for line in lines:
                for x1,y1,x2,y2 in line:
                    cv2.line(framelines,(x1,y1),(x2,y2),(0,255,0),2)
            utils.plot(framelines, "Line Detection")




            writer.write(frame)

