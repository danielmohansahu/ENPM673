"""Top level functions used for Lane Detection
"""
import os
import numpy as np
import cv2
from . import lane, utils, file_utils, process

"""Process and Plot a single image.
"""
def process_image(frame, intrinsics, left_lane, right_lane, debug=False):
    
    if debug:
        utils.plot(frame, "Input Frame")
    result = frame.copy()
    
    # undistort image
    K,D = intrinsics
    frame = process.rectify(frame, K, D)
    if debug:
        utils.plot(frame, "Rectified Image")

    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if debug:
        utils.plot(gray, "Grayscale")
    
    # convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    if debug:
        utils.plot(hsv, "HSV Image")

    # extract yellow and white pixels (expected lane colors)
    frame = process.extract_yw(gray,hsv)
    if debug:
        utils.plot(frame, "Yellow and White")

    # denoise the image
    frame = cv2.GaussianBlur(frame,(7,7),cv2.BORDER_DEFAULT)
    if debug:
        utils.plot(frame, "Filtered")
    
    # edge detection
    frame = cv2.Canny(frame, 50, 125)
    if debug:
        utils.plot(frame, "Detected Edges")

    # ROI out upper half of image
    Y,X = frame.shape
    vertices=np.array([[0,Y//2],[X,Y//2],[X,Y],[0,Y]])
    frame = process.roi(frame, vertices)
    if debug:
        utils.plot(frame, "ROI")
    
    # hough line detection
    lines, hough = process.hough_lines(frame) 
    if debug:
        utils.plot(hough, "Hough Lines")

    # Lane Filtering / Tracking
    ll = left_lane.get_best_match(lines[:,0])
    rl = right_lane.get_best_match(lines[:,0])
    if rl:
        cv2.line(result,(rl[0],rl[1]),(rl[2],rl[3]),(0,0,255),2)
    if ll:
        cv2.line(result,(ll[0],ll[1]),(ll[2],ll[3]),(0,0,255),2)
    if debug: 
        utils.plot(result, "Lanes")

    return result

"""Process and write to a video.
"""
def process_video(videofile, 
                  intrinsics, 
                  left_lane, 
                  right_lane, 
                  debug=False, 
                  output_prefix="processed_"):
    # initialize our video IO
    vidgen = file_utils.VidGenerator(videofile, debug)
    output_file = "processed_" + os.path.basename(videofile)
    video_writer = file_utils.VidWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), vidgen.fps, vidgen.size, isColor=True)

    # step through each frame and process
    with video_writer as writer:
        for ret,frame in vidgen:
            result = process_image(frame, intrinsics, left_lane, right_lane, debug)
            writer.write(result)


