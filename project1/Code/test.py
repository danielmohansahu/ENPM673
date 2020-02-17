#!/usr/bin/env python3

import time
import code
import numpy as np
import cv2
from custom import file_utils
from custom.tracking import get_corners, get_homography, to_homogeneous, get_contour_x_bounds, replace_section, warp_image

from custom.detection import ARDetector

# TEST_FILE = "../Data/Video_dataset/Tag0.mp4"
TEST_FILE = "../Data/Video_dataset/Tag1.mp4"
AR_FILE = "../Data/reference_images/ref_marker.png"
TEMPLATE_FILE = "../Data/reference_images/Lena.png"

if __name__ == "__main__":
    
    # set debugging on (this will stop at every frame!)
    ARDetector.debug(True)
    
    # get template file
    template = file_utils.imread(TEMPLATE_FILE)
    template_corners = get_corners(template)

    for ret,frame in file_utils.VidGenerator(TEST_FILE):
        st = time.time()
        detector = ARDetector(frame)
        detections = detector.detect()
        print("{:3f}s to detect.".format(time.time()-st))

        for corners, orientation in detections:
            # get the mapping from template pixels to image pixels

            st = time.time()
            H = get_homography(corners, template_corners, 0)
            print("{:3f}s to get homography.".format(time.time()-st))
            
            st = time.time()
            warped = warp_image(template, H, frame.shape)
            print("{:3f}s to warp.".format(time.time()-st))

            st = time.time()
            test = replace_section(frame, warped, corners)
            print("{:3f}s to replace.".format(time.time()-st))
            
            ARDetector.plot(test)
            # ARDetector.plot(frame)
            # ARDetector.plot(warped)
            

    code.interact(local=locals())
