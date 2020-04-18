#!/usr/bin/env python3

import os
import glob
import cv2
import numpy as np
from scipy import ndimage
from custom.LucasKanade import LucasKanade
from custom import file_utils

# TEMPLATE_BBOX=[269,75,34,64]
# FILEPATH="../data/Bolt2/img/"
# VIDEOFILE="Bolt2.mp4"

# TEMPLATE_BBOX=[70,51,107,87]
# FILEPATH="../data/Car4/img/"
# VIDEOFILE="Car4.mp4"

TEMPLATE_BBOX=[160,83,56,65]
FILEPATH="../data/DragonBaby/img/"
VIDEOFILE="DragonBaby.mp4"

def draw_rectangle(frame, bb, affine):
    """Perform the given affine transformation on the bounding box and draw it on the frame.
    """
    # turn bounding box into points
    pts = np.array([
        [bb[0],bb[1],1],
        [bb[0]+bb[2],bb[1],1],
        [bb[0]+bb[2],bb[1]+bb[3],1],
        [bb[0],bb[1]+bb[3],1]],
        dtype=np.float32)

    # generate our transformation matrix (inverted, template->image)
    M = np.vstack((affine + np.array([[1,0,0],[0,1,0]]), np.array([0,0,1]))) 
    Minv = np.linalg.inv(M)

    # apply affine transform
    Minv = np.linalg.inv(M)
    rot_pts = []
    for pt in pts:
        rot = np.dot(M,pt)[:2]
        rot_pts.append(rot)

    # draw rectangle
    result = cv2.polylines(frame, np.array([rot_pts],np.int32), True, (0,0,255))
    result = cv2.transpose(result)
    result = frame

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
    video_writer = file_utils.VidWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 5, (template.shape[1],template.shape[0]), isColor=False)

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

            if p is not None:
                # draw rectangle
                result = draw_rectangle(frame, TEMPLATE_BBOX, p)
            else:
                # just draw an empty image
                result = frame.copy()
                continue

            writer.write(result)
