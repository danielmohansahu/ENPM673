#!/usr/bin/env python3

import os
import sys
import time
import glob
import argparse
import cv2
import numpy as np
from custom.LucasKanade import LucasKanade
from custom import file_utils

DATASETS = {
    'bolt': {
        'bbox': [269,75,34,64],
        'filepath': "../data/Bolt2/img/",
        'videofile': "Bolt2.mp4"},
    'car': {
        'bbox': [70,51,107,87],
        'filepath': "../data/Car4/img/",
        'videofile': "Car4.mp4"},
    'dragonbaby': {
        'bbox': [160,83,56,65],
        'filepath': "../data/DragonBaby/img/",
        'videofile': "DragonBaby.mp4"}
}

def parse_args():
    """ Command line Arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, help="Dataset to run (choose from {}".format(DATASETS.keys()))
    parser.add_argument("-p", "--prefix", default="processed_", help="Prefix to prepend to output video name")
    parser.add_argument("-e", "--epsilon", type=float, default="0.01", help="Maximum norm of transform delta to determine convergence.")
    parser.add_argument("-s", "--sigma", type=float, default=10, help="Huber Loss parameter.")
    parser.add_argument("-m", "--max-count", type=int, default=400, help="Maximum number of iterations per frame.")
    parser.add_argument("-a", "--avg-frames", type=int, default=5, help="Number of previous frames to average as template image.")
    args = parser.parse_args()
    if args.dataset not in DATASETS.keys():
        raise RuntimeError("Invalid dataset. Got {}, expected {}".format(args.dataset, DATASETS.keys()))
    return args

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

    # apply affine transform
    rot_pts = []
    for pt in pts:
        rot = np.dot(M,pt)[:2]
        rot_pts.append(rot)

    # draw min bounding rectangle
    ctr = np.array(rot_pts, dtype = np.float32)
    x,y,w,h = cv2.boundingRect(ctr)
    result = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    # draw affine rectangle (for reference)
    result = cv2.polylines(frame, np.array([rot_pts],np.int32), True, (0,0,255))

    return result

if __name__ == "__main__":
    # parse args
    args = parse_args()
    filepath = DATASETS[args.dataset]['filepath']
    bbox = DATASETS[args.dataset]['bbox']
    videofile = DATASETS[args.dataset]['videofile']

    # get all image files (in order)
    images = glob.glob(filepath+"*.jpg")
    images.sort()

    # initialize tracker with initial frame and bounding box
    template = cv2.imread(images[0],0)
    lk = LucasKanade(template,
                     bounding_box=bbox,
                     epsilon=args.epsilon,
                     sigma=args.sigma,
                     max_count=args.max_count,
                     avg_frames=args.avg_frames)

    # generate video reader / writer objects
    output_file = args.prefix + os.path.basename(videofile)
    video_writer = file_utils.VidWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 5, (template.shape[1],template.shape[0]), isColor=False)

    # step through each frame and process
    p = np.zeros((2,3),dtype=np.float32)
    with video_writer as writer:
        # draw first image
        frame = cv2.imread(images[0],0)
        images.pop(0)
        result = draw_rectangle(frame, bbox, p)
        writer.write(result)

        # initialize metavariables
        count = 1
        failures = 0
        st = time.time()

        for image in images:
            # update status bar
            count += 1
            bar = ("="*int(20*count/len(images))).ljust(20)
            sys.stdout.write('\r')
            sys.stdout.write("[{}] {} {:.0f}s, {} bad".format(bar, "/".join(image.split("/")[-3:]), time.time()-st, str(failures)))
            sys.stdout.flush()

            # read frame
            frame = cv2.imread(image,0)

            # estimate tracked transform
            success, p = lk.estimate(frame)

            if p is not None:
                # draw rectangle
                result = draw_rectangle(frame, bbox, p)
            else:
                # just draw an empty image
                result = frame.copy()

            # update failure / non-converged count
            if not success:
                failures += 1

            writer.write(result)
