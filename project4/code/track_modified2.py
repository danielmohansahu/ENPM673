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

truth_data = np.loadtxt("../data/DragonBaby/groundtruth_rect.txt", dtype = int, delimiter = ',')

def draw_rect_grd(frame, bb, p1=[[1,0,0],[0,1,0]]):
    print("bb", bb)
    # turn bounding box into points
    pts = np.array([
        [bb[0],bb[1]],
        [bb[0]+bb[2],bb[1]],
        [bb[0]+bb[2],bb[1]+bb[3]],
        [bb[0],bb[1]+bb[3]]],
        dtype=np.float32)
    print("grd pts", pts)
    result = cv2.polylines(frame, np.int32([pts]), True, (0,255,0))
    return result

def draw_rectangle(frame, bb, affine):
    """Perform the given affine transformation on the bounding box and draw it on the frame.
    """
    
    print("bb", bb)
    # turn bounding box into points
    pts = np.array([
        [bb[0],bb[1]],
        [bb[0]+bb[2],bb[1]],
        [bb[0]+bb[2],bb[1]+bb[3]],
        [bb[0],bb[1]+bb[3]]],
        dtype=np.float32)
    A = affine[:, 0:2]
    B = affine[:, 2]
    
    rot_pts = pts@A + B
    print("rot", rot_pts)
   
    corner_pt = np.array([rot_pts[0,:]], dtype = np.float32)
    #print('type corner', type(corner_pt), type(rot_pts), type(pts))
    #print("corner_pt",corner_pt[0][0])
    new_box = np.array(
               [[corner_pt[0][0], corner_pt[0][1]], 
               [corner_pt[0][0] + bb[2], bb[1]], 
               [corner_pt[0][0] + bb[2], corner_pt[0][1] + bb[3]], 
               [corner_pt[0][0], corner_pt[0][1] + bb[3] ] ],
               dtype = np.float32)
    
    print('new_box', new_box)
    # apply affine transform
    #rot_pts = cv2.warpAffine(pts, affine, (pts.shape[1],pts.shape[0]))
    
    # draw rectangle
    result = cv2.polylines(frame, np.int32([new_box]), True, (255,255,255))
    return result, bb

if __name__ == "__main__":
    # get all image files (in order)
    images = glob.glob(FILEPATH+"*.jpg")
    images.sort()

    # initialize tracker with initial frame and bounding box
    template = cv2.imread(images[0],0)
    #cv2.imshow("first", template)
    #cv2.waitKey(0)
    lk = LucasKanade(template,TEMPLATE_BBOX)
    #lk.template[bb[0]:bb[0]+bb[2],bb[1]:bb[1]+bb[3]]
    # generate video reader / writer objects
    output_file = "processed_" + os.path.basename(VIDEOFILE)
    video_writer = file_utils.VidWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 5, (640,360), isColor=False)
    count = 0
    # step through each frame and process
    p = np.array([[1,0,0],[0,1,0]],dtype=np.float32)
    with video_writer as writer:
        # draw first image
        frame = cv2.imread(images[0],0)
        images.pop(0)
        result, bb = draw_rectangle(frame, TEMPLATE_BBOX, p)
        cv2.imshow("bb",result)
        cv2.waitKey(0)
        writer.write(result)

        for image in images:
            print("Processing frame: {}".format(image))
            frame = cv2.imread(image,0)

            # estimate tracked transform
            p = lk.estimate(frame)
            print(p)
            count +=1
            # draw rectangle
            result, bb = draw_rectangle(frame, TEMPLATE_BBOX, p)
            #print(truth_data)
            ground_truth = draw_rect_grd(frame, list(truth_data[count]), p )
            cv2.imshow("processed", ground_truth)
            cv2.waitKey(0)
            writer.write(frame)