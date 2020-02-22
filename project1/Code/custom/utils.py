"""Collection of miscellaneous useful functions.
"""
import time
import numpy as np

def to_homogeneous(points):
    """Convert a given list of np arrays to Homogeneous coordinates.
    """
    homogeneous = []
    for point in points:
        homogeneous.append(point.flatten().tolist() + [1])
        return np.array(homogeneous)

def get_corners(image):
    """Get the outer corners of the given image.
    """
    y,x,_ = image.shape
    corners = np.array([[[0,0]],[[x,0]],[[x,y]],[[0,y]]])
    return corners

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
