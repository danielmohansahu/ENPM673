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

VIDEO_FILE = "Tag0.mp4"
TAG_FILE = "ref_marker.png"
REFERENCE_FILE = "Lena.png"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=VIDEO_FILE, help="Input video containing fiducial images to process.")
    parser.add_argument("--tag_file", type=str, default=TAG_FILE, help="Image of the fiducial to detect.")
    parser.add_argument("--reference_file", type=str, default=REFERENCE_FILE, help="Input picture to replace the detected tag.")
    parser.add_argument("--verbosity", type=int, default=1, help="Set verbosity level (0 is none, 1 is console output, 2 is images).")
    return parser.parse_args()

if __name__ == "__main__":
    # parse command line args
    args = parse_args()
    
    # get template file (lena) and reference tag file
    template = file_utils.imread(args.reference_file)
    reference_tag = file_utils.imread(args.tag_file,0)

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
                proj_mat = projection_matrix(homography)
                frame = cube(proj_mat, frame)
                if args.verbosity > 1:
                    detector.plot(frame, "cube")
           
            writer.write(frame)
            if args.verbosity:
                ctime = time.time()
                print("Found ids {} in frame #{}/{} in {:.3f}s ({:.3f}s total)".format(ids, frame_count, vidgen.frame_count, ctime-frame_start, ctime-process_start))


