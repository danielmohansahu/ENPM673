"""Collection of miscellaneous useful functions.
"""
import cv2
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
