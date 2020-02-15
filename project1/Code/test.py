#!/usr/bin/env python3

import code
import numpy as np
import cv2
from custom import file_utils
from custom.tracking import get_corners, get_homography, to_homogeneous, get_contour_x_bounds, replace_section

from custom.detection import ARDetector

TEST_FILE = "../Data/Video_dataset/Tag0.mp4"
AR_FILE = "../Data/reference_images/ref_marker.png"
TEMPLATE_FILE = "../Data/reference_images/Lena.png"

if __name__ == "__main__":
    
    # set debugging on (this will stop at every frame!)
    ARDetector.debug(False)
    
    # get template file
    template = file_utils.imread(TEMPLATE_FILE)
    template_corners = get_corners(template)

    for ret,frame in file_utils.VidGenerator(TEST_FILE):
        detector = ARDetector(frame)
        detections = detector.detect()

        for corners, orientation in detections:
            # get the mapping from template pixels to image pixels
            H = get_homography(corners, template_corners, 0)
          
            warped = cv2.warpPerspective(template, H, (frame.shape[1],frame.shape[0]))

            # test = replace_section(frame, warped, corners)
            ARDetector.plot(warped)
            

    code.interact(local=locals())
