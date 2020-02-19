#!/usr/bin/env python3

import time
import code
import argparse
import numpy as np
import cv2

from custom import file_utils
from custom.tracking import ARTracker
from custom.detection import ARDetector

TEST_FILE = "../Data/Video_dataset/Tag1.mp4"
AR_FILE = "../Data/reference_images/ref_marker.png"
TEMPLATE_FILE = "../Data/reference_images/Lena.png"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=TEST_FILE, help="Input video containing fiducial information.")
    parser.add_argument("--debug", type=bool, default=False, help="Plot processed images (and timing)")
    return parser.parse_args()

if __name__ == "__main__":
    # parse command line args
    args = parse_args()
    
    # get template file (lena) and reference tag file
    template = file_utils.imread(TEMPLATE_FILE)
    reference_tag = file_utils.imread(AR_FILE, grayscale=True)

    # initialize tracker and set class debugging
    tracker = ARTracker(template)
    ARDetector.debug(True)
    ARTracker.debug(True)

    # process every frame in the given video
    for ret,frame in file_utils.VidGenerator(args.video):
        st = time.time()
        detector = ARDetector(frame, reference_tag)
        detections = detector.detect()

        for corners, orientation in detections:
            tracker.track(frame, corners) 

    code.interact(local=locals())
