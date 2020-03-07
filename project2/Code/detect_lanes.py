#!/usr/bin/env python3

import os
import argparse
import cv2
import numpy as np
from custom import utils, file_utils, pre_process, lines

# defaults (for testing)
CALIBRATION="../Data/data_1/camera_params.yaml"
VIDEO="../Data/data_1/data_1.mp4"
# manually selected lane points
LANE_POINTS=[[0,422],[589,256],[694,244],[840,510]]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-V","--video",type=str,default=VIDEO,help="Video file to process")
    parser.add_argument("-v","--verbosity",type=int,default=1,help="Verbosity")
    parser.add_argument("-c","--calibration",type=str,default=CALIBRATION,help="Yaml file containing camera intrinsics.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # get camera calibration
    K,D = file_utils.load_params(args.calibration)

    # initialize our video IO
    vidgen = file_utils.VidGenerator(args.video, args.verbosity)
    output_file = "processed_" + os.path.basename(args.video)
    video_writer = file_utils.VidWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), vidgen.fps, vidgen.size, isColor=True)

    # compute homography (from manual points)
    subsect = utils.get_subsection(vidgen.size,400,250)
    H,_ = cv2.findHomography(np.array(LANE_POINTS),subsect) 

    # step through each frame and process
    with video_writer as writer:
        for ret,frame in vidgen:
            # utils.plot(frame, "Frame")
            
            # undistort image
            rect = pre_process.rectify(frame, K, D)
            # utils.plot(rect, "Rect")

            # convert to grayscale
            gray = cv2.cvtColor(rect, cv2.COLOR_RGB2GRAY)
            # utils.plot(gray, "Grayscale")
            
            # convert to HSV
            hsv = cv2.cvtColor(rect, cv2.COLOR_RGB2HSV)
            # utils.plot(hsv, "HSV")

            lower_yellow = np.array([20, 100, 100], dtype = "uint8")
            upper_yellow = np.array([30, 255, 255], dtype="uint8")
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask_white = cv2.inRange(gray, 200, 255)
            mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
            mask_yw_image = cv2.bitwise_and(gray, mask_yw)
            # utils.plot(mask_yw_image, "YellowAndWhite")

            # warp perspective (convert lanes to "vertical")
            # warped = cv2.warpPerspective(mask_yw_image, H, dsize=vidgen.size)
            # utils.plot(warped, "Warped")
            # warped = mask_yw_image.copy()

            # denoise the image
            blur = cv2.GaussianBlur(mask_yw_image,(7,7),cv2.BORDER_DEFAULT)
            # utils.plot(blur, "Blur")
            
            # threshold
            # ret, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
            # utils.plot(thresh, "thresh")
           
            # edge detection
            edges = cv2.Canny(blur, 50, 125)
            # utils.plot(edges, "Edge Detection")

            # ROI
            mask = np.zeros_like(edges)
            vertices=np.array([[0,256],[1392,256],[1392,512],[0,512]])
            cv2.fillPoly(mask, [vertices], 255)
            roi_image = cv2.bitwise_and(edges, mask)
            utils.plot(roi_image, "ROI")
            
            # hough lines
            rho = 1
            theta = np.pi/40
            threshold = 30
            min_line_len = 100
            max_line_gap = 250
            lines = cv2.HoughLinesP(
                    roi_image, 
                    rho, 
                    theta, 
                    threshold, 
                    np.array([]), 
                    minLineLength=min_line_len, 
                    maxLineGap=max_line_gap)
            line_img = np.zeros(frame.shape, dtype=np.uint8)
           
            for x1,y1,x2,y2 in lines[:,0]:
                cv2.line(line_img,(x1,y1),(x2,y2),(0,0,255),2)
            
            utils.plot(line_img, "Hough")


            
            result = line_img.copy()

            # fit lines (via column histogram method)
            # result,fits = lines.polyfit(result)
            # utils.plot(result, "Detections")

            writer.write(result)

