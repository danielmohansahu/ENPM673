"""Functions used to improve the quality of a given frame.
"""

import numpy as np
import cv2

def gamma_correct(frame, gamma=2.0):
    """ Apply gamma correction to frame
    https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    """
    table = []
    for i in np.arange(256):
        table.append(((i / 255.0) ** (1/gamma)) * 255)
    table = np.array(table,dtype="uint8")
    return cv2.LUT(frame, table)

def rectify(frame, camera_matrix, distortion):
    """Rectify the given image.

    This is just a thin wrapper around cv2.undistortPoints
    """
    frame = cv2.undistort(frame, camera_matrix, distortion)
    return frame

def extract_yw(gray, hsv, y_lower=[20,100,100], y_upper=[100,255,255], w_lower=200):
    """Extract Yellow and White pixels from the given HSV/Gray frames.
    """
    # construct yellow mask
    mask_yellow = cv2.inRange(hsv, 
                              np.array(y_lower,dtype="uint8"), 
                              np.array(y_upper,dtype="uint8"))
    # construct white mask
    mask_white = cv2.inRange(gray, w_lower, 255)

    # full mask:
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    
    # use bitwise logic to get the Yellow or White frames:
    result = cv2.bitwise_and(gray, mask_yw)
    return result

def roi(frame, vertices):
    """Set everything ourside the given ROI to 0."""
    # create Zero mask and fill ROI with values of 1
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [vertices], 255)
    result = cv2.bitwise_and(frame, mask)
    return result

def hough_lines(frame, rho=1, theta=np.pi/40, threshold=30, min_line_len=100, max_line_gap=250):
    """Extract Hough Lines
    """
    result = np.zeros((frame.shape[0],frame.shape[1],3), dtype=np.uint8)
    lines = cv2.HoughLinesP(
            frame, 
            rho, 
            theta, 
            threshold, 
            np.array([]), 
            minLineLength=min_line_len, 
            maxLineGap=max_line_gap)

    # draw lines
    for x1,y1,x2,y2 in lines[:,0]:
        cv2.line(result,(x1,y1),(x2,y2),(0,0,255),2)

    return lines, result

def sharpen(frame, kernel_size=(5,5), sigma=1.0, amount=1.0):
    """Sharpen the high frequency components of the given frame.

    https://stackoverflow.com/questions/4993082/how-to-sharpen-an-image-in-opencv
    """
    # smooth input image
    blurred = cv2.GaussianBlur(frame, kernel_size, sigma)

    # calculate sharpened image
    sharpened = float(amount + 1) * frame - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    return sharpened


def equalize(frame):
    """Apply histogram equalization to the given frame.
    
    This uses an adaptive histogram equalization algorithm,
    to handle the contrast between very bright and very
    dark regions.
    """
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    frame = clahe.apply(frame)

    return frame
