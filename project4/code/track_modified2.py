#!/usr/bin/env python3

import os
import sys
import time
import glob
import argparse
import cv2
import numpy as np
from custom.LucasKanade_cleanup import LucasKanade
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, help="Dataset to run (choose from {}".format(DATASETS.keys()))
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
    result = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
    bbox_new = [x,y,w,h]
    # draw affine rectangle (for reference)
    result = cv2.polylines(frame, np.array([rot_pts],np.int32), True, (0,0,255))
    M = cv2.moments(ctr)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
    
    return result, bbox_new

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
    template1 = template

    # generate video reader / writer objects
    output_file = "processed_" + os.path.basename(videofile)
    video_writer = file_utils.VidWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 5, (template.shape[1],template.shape[0]), isColor=False)

    # step through each frame and process
    p = np.zeros((2,3),dtype=np.float32)
    with video_writer as writer:
        lk = LucasKanade(template,bbox)
        print("bbox", bbox)
        # draw first image
        frame = cv2.imread(images[0],0)
        images.pop(0)
        result, bbox_new = draw_rectangle(frame, bbox, p)
        print("bbox_new", bbox_new)
        writer.write(result)

        # initialize metavariables
        count = 1
        failures = 0
        st = time.time()
        frame_avg =  np.float32(template)
        template_avg = np.zeros((270,480), dtype = float)
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
                result, bbox_new = draw_rectangle(frame, bbox, p)
            else:
                # just draw an empty image
                result = frame.copy()

            # update failure / non-converged count
            if not success:
                failures += 1
            
            writer.write(result)
            if count%5 != 0:
                c = float(count%5)
                frame_avg = np.float32(frame)
                #print(type(frame_avg))
                #print(type(template1))
                template_avg += frame_avg/c
                template = cv2.normalize(template_avg, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                #cv2.imshow("ave template", template)
                #cv2.waitKey(0)
                
            elif count%5 == 0:
                lk = LucasKanade(template, bbox_new)
                
            
            
