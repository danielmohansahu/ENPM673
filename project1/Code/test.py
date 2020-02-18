#!/usr/bin/env python3

import time
import code
import numpy as np
import cv2

from custom import file_utils
from custom.tracking import ARTracker
from custom.detection import ARDetector

# TEST_FILE = "../Data/Video_dataset/Tag0.mp4"
TEST_FILE = "../Data/Video_dataset/Tag1.mp4"
AR_FILE = "../Data/reference_images/ref_marker.png"
TEMPLATE_FILE = "../Data/reference_images/Lena.png"

if __name__ == "__main__":
    
    # set debugging on (this will stop at every frame!)
    ARDetector.debug(True)
    ARTracker.debug(True)
    
    # get template file
    template = file_utils.imread(TEMPLATE_FILE)
    tracker = ARTracker(template)

    for ret,frame in file_utils.VidGenerator(TEST_FILE):
        st = time.time()
        detector = ARDetector(frame)
        detections = detector.detect()

        for corners, orientation in detections:
            tracker.track(frame, corners) 

    code.interact(local=locals())
