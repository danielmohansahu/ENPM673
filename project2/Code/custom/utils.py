"""Collection of miscellaneous useful functions.
"""
import cv2
import numpy as np
import time

# global variables
STOP_PLOTTING=False

class Timer(object):
    def __init__(self, description, verbosity=0):
        self.description = description
        self.verbosity = verbosity
    def __enter__(self):
        if self.verbosity:
            self.start = time.time()
    def __exit__(self, type, value, traceback):
        if self.verbosity > 1:
            self.end = time.time()
            print(f"{self.description}: {self.end - self.start}")

def get_subsection(size,p_x,p_y):
    """Return a rectangular contour centered around the given frame size.

    This returns a set of points ordered ClockWise, starting from
    the bottom left point.

    Args:
        size: [X,Y] dimension size
        p_x: X dimension padding (rectangle width)
        p_y: Y dimension padding (rectangle height)
    """ 
    c_x = size[0]//2
    c_y = size[1]//2
    result = [[c_x-p_x,c_y+p_y],[c_x-p_x,c_y-p_y],[c_x+p_x,c_y-p_y],[c_x+p_x,c_y+p_y]]
    return np.array(result)

def plot(frame, text="frame"):
    """Plot a given frame.
    """
    global STOP_PLOTTING
    if STOP_PLOTTING:
        return

    # convenience function for debugging
    cv2.namedWindow(text, cv2.WINDOW_NORMAL)
    cv2.imshow(text, frame)

    key = cv2.waitKey(0)
    if key == ord('q'):
        # stop debugging on 'q'
        STOP_PLOTTING=True

    cv2.destroyAllWindows()
