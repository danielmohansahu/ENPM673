#!/usr/bin/env python3
import os
import time
import code
import argparse
import numpy as np
import cv2

from custom import file_utils
from custom.tracking import ARTracker
from custom.detection import ARDetector
from custom.projection import cube, draw, projection_matrix 

TEST_FILE = "../Data/Video_dataset/Tag0.mp4"
AR_FILE = "../Data/reference_images/ref_marker.png"
TEMPLATE_FILE = "../Data/reference_images/Lena.png"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=TEST_FILE, help="Input video containing fiducial information.")
    parser.add_argument("--verbosity", type=int, default=1, help="Set verbosity level (0 is none, 1 is console output, 2 is images).")
    return parser.parse_args()

if __name__ == "__main__":
    # parse command line args
    args = parse_args()
    
    # get template file (lena) and reference tag file
    template = file_utils.imread(TEMPLATE_FILE)
    reference_tag = file_utils.imread(AR_FILE,0)

    # initialize tracker and set class debugging
    tracker = ARTracker(template)
    ARDetector.debug(args.verbosity)
    ARTracker.debug(args.verbosity)

    # initialize our video IO
    vidgen = file_utils.VidGenerator(args.video)
    output_file = "processed_" + os.path.basename(args.video)
    video_writer = file_utils.VidWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), vidgen.fps, vidgen.size)
    
    # process every frame in the given video
    process_start = time.time()
    frame_count = 0
    with video_writer as writer:
        for ret,frame in vidgen:
            # metadata
            frame_start = time.time()
            frame_count += 1
            # Create a black image
            #black = np.zeros((1980, 1080), np.uint8)
            detector = ARDetector(frame, reference_tag)
            detections, ids = detector.detect()

            for corners in detections:
                frame, homography = tracker.track(frame, corners)
            '''
            question here - I put this in because the code broke sometimes, same as below
            '''    
            if detections != []:
                print('contours', detections[0])
            # Create the cube
            
            s = 512
            source = np.array([[0,0], [s,0], [s,s],[0,s]])
      
            
            # Maybe some frames aren't getting a detection?? Stops program when []
            '''
            question here - I put this in because the code broke sometimes
            '''     
            pts = np.array(detections)
            if pts != []:
                pts_im = pts.reshape(4,2)
               
            
            H = ARTracker.get_homography( pts_im, source)
            proj_mat = projection_matrix(H)
            
            image = cube(proj_mat, frame)
            cv2.imshow('cube', image)
            cv2.waitKey(1)
           
           
            writer.write(frame)
            if args.verbosity:
                ctime = time.time()
                #print("Found ids {} in frame #{}/{} in {:.3f}s ({:.3f}s total".format(ids, frame_count, vidgen.frame_count, ctime-frame_start, ctime-process_start))


