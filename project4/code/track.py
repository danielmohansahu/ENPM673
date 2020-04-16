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

def draw_rectangle(frame, bb, affine):
    """Perform the given affine transformation on the bounding box and draw it on the frame.
    """
    # turn bounding box into points
    pts = np.array([
        [bb[0],bb[1]],
        [bb[0]+bb[2],bb[1]],
        [bb[0]+bb[2],bb[1]+bb[3]],
        [bb[0],bb[1]+bb[3]]],
        dtype=np.float32)

    # apply affine transform
    M = np.vstack((affine + np.array([[1,0,0],[0,1,0]]), np.array([0,0,1]))) 
    M = np.linalg.inv(M)
    rot_pts = cv2.perspectiveTransform(pts[None,:,:], M)
    
    # draw rectangle
    result = cv2.polylines(frame, np.int32([rot_pts]), True, (0,0,255))
    return result

if __name__ == "__main__":
    # get all image files (in order)
    images = glob.glob(FILEPATH+"*.jpg")
    images.sort()

    # initialize tracker with initial frame and bounding box
    template = cv2.imread(images[0],0)
    lk = LucasKanade(template,TEMPLATE_BBOX)

    # generate video reader / writer objects
    output_file = "processed_" + os.path.basename(VIDEOFILE)
    video_writer = file_utils.VidWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 5, (640,360), isColor=False)

    # step through each frame and process
    p = np.zeros((2,3),dtype=np.float32)
    with video_writer as writer:
        # draw first image
        frame = cv2.imread(images[0],0)
        images.pop(0)
        result = draw_rectangle(frame, TEMPLATE_BBOX, p)

        writer.write(result)

        for image in images:
            print("Processing frame: {}".format(image))
            frame = cv2.imread(image,0)

            # estimate tracked transform
            p = lk.estimate(frame)
            print(p)

            # draw rectangle
            result = draw_rectangle(frame, TEMPLATE_BBOX, p)

            writer.write(frame)
