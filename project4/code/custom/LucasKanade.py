"""Python implementation of the Lucas-Kanade algorithm for template tracking via Affine Transformation.
"""

import numpy as np
import cv2

class LucasKanade:

    def __init__(self, template, bbox):
        """Initialize the Lucas-Kanade algorithm.

        Args:
            template:   Image containing the template to track.
            bbox:       Bounding box of the template.
        """

        # we only allow affine transformations for our warp function
        W = lambda p1,p2,p3,p4,p5,p6: np.array([[1+p1,p3,p5],[p2,1+p4,p6]]) 

    def estimate(self, frame):
        """Estimate the warp parameters that best fit the given frame.

        Note that this implicitly assumes that frames are supplied
        in sequence. This method uses the previously solved for warp 
        parameters of (presumably) the previous frame in the same sequence.
        """
        p = None

        return p
