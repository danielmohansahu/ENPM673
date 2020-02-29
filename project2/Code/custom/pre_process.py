"""Functions used to improve the quality of a given frame.
"""

import numpy as np
import cv2

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
