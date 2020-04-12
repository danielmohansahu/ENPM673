#!/usr/bin/env python3

import os
import glob
import cv2
import numpy as np
from custom.LucasKanade import LucasKanade
from custom import file_utils

TEMPLATE_BBOX=[160,83,56,65]
FILEPATH="../data/DragonBaby/img/"
VIDEOFILE="DragonBaby.mp4"

if __name__ == "__main__":
    # get all image files (in order)
    images = glob.glob(FILEPATH+"*.jpg")
    images.sort()

    # initialize tracker with initial frame and bounding box
    template = cv2.imread(images[0],0)
    lk = LucasKanade(template,TEMPLATE_BBOX)

    # generate video reader / writer objects
    output_file = "processed_" + os.path.basename(VIDEOFILE)
    video_writer = file_utils.VidWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640,360), isColor=True)

    # step through each frame and process
    with video_writer as writer:
        for image in images:
            print("Processing frame: {}".format(image))
            frame = cv2.imread(image,0)

            p = lk.estimate(frame)
            print(p)

            # writer.write(frame)
