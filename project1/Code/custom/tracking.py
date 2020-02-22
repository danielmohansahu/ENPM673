"""Classes and scripts related to Homography and Tracking.
"""

import numpy as np
import cv2
from .utils import Timer, get_corners

class ARTracker:
    """Class containing functions for AR tracking / manipulation.
    """
    __debug = False

    def __init__(self, template):
        # template image to use in AR replacement
        self._template = template
        self._template_corners = get_corners(self._template)
   
    def track(self, frame, ar_contour):
        # compute homography
        with Timer("\thomography", self.__debug):
            H = self.get_homography(ar_contour, self._template_corners)
        
        # warp template image
        with Timer("\twarping", self.__debug):
            warped = self.warp_image(self._template, H, frame.shape)
        if self.__debug:
            self.plot(warped, "warped")

        # replace AR tag with warped image
        with Timer("\treplace", self.__debug):
            replaced = self.replace_section(frame, warped, ar_contour)
        if self.__debug:
            self.plot(replaced, "replaced")

        return replaced
    #------------------ STATIC API FUNCTIONS ------------------#

    @staticmethod
    def get_homography(template_corners, corners):
        """Compute homography between sets of corners.
        """
        # sanity checks
        if len(corners) != len(template_corners):
            raise RuntimeError("Given different sized corner arrays for homography; {} vs {}".format(template_corners,corners))
    
        # compute set of homography linear equations:
        A = np.zeros([2*len(corners), 9])
        for i in range(len(corners)):
            x1w,y1w = corners[i].flatten().tolist()
            x1c,y1c = template_corners[i].flatten().tolist()
            A[2*i]   = np.array([x1w,y1w,1,0,0,0,-x1c*x1w,-x1c*y1w,-x1c])
            A[2*i+1] = np.array([0,0,0,x1w,y1w,1,-y1c*x1w,-y1c*y1w,-y1c])
    
        # calculate SVD
        U,D,V = np.linalg.svd(A)
    
        # select smallest eigenvalue and normalize
        # note that we're selecting a row, since SVD returns the 
        #   transpose of V
        H = V[-1,:]/V[-1,-1]
        H.resize([3,3])
    
        return H

    @staticmethod
    def warp_image(frame, H, output_shape):
        """Warp the given image with the given homography matrix.
        """
        dst = np.zeros(output_shape)
    
        # construct homogeneous form of indices
        size_x, size_y = frame.shape[:-1]
        dst_hom = np.array([
            np.array([[i]*size_x for i in range(size_y)]).flatten(),
            [*range(size_x)]*size_y,
            [1]*size_x*size_y
            ])
    
        # homographize and normalize:
        homography = H@dst_hom
        x_f = np.array(homography[0,:]/homography[2,:],dtype=np.int)
        y_f = np.array(homography[1,:]/homography[2,:],dtype=np.int)
    
        for i in range(len(x_f)):
            dst[y_f[i], x_f[i],:] = frame[dst_hom[1,i],dst_hom[0,i],:]
    
        return dst
    
    @classmethod
    def plot(cls, frame, text="frame"):
        # convenience function for debugging
        cv2.namedWindow(text, cv2.WINDOW_NORMAL)
        cv2.imshow(text, frame)

        key = cv2.waitKey(0)
        if key == ord('q'):
            # stop debugging on 'q'
            cls.debug(False)
        cv2.destroyAllWindows()

    @classmethod
    def debug(cls, on_or_off):
        cls.__debug = on_or_off

    def replace_section(self, src, replacement, contour):
        """Replace the contour in src with the same contour in dst.
        Images must be the same size.
        """
        if src.shape != replacement.shape:
            raise RuntimeError("Unable to process images of different shapes. {} vs {}".format(src.shape, replacement.shape))
    
        # initialize result image
        result = src.copy()
    
        # initialize some loop variables
        size_y, size_x = src.shape[:-1]
        X_c = [c[0][0] for c in contour]    # contour X value list
        Y_c = [c[0][1] for c in contour]    # contour Y value list
    
        for y_pix in range(size_y):
            # find out which columns of the given row should be replaced, and replace.
            bounds = self.get_contour_x_bounds(y_pix,X_c,Y_c)
            if bounds:
                result[y_pix,bounds[0]:bounds[1]] = replacement[y_pix,bounds[0]:bounds[1]]
    
        return result
    
    def get_contour_x_bounds(self, y_val, X, Y):
        """Calculate the minimum and maximum X that are still within
        the given contour (X,Y) for the given Y value.
    
        @TODO handle edge cases (on a corner, on a line) better
        """
        # return empty range if we're above or below the contour
        if y_val < min(Y) or y_val > max(Y):
            return None
        
        # general case; find the two lines our row intersects
        Y_closed = Y + [Y[0]] # close for easier looping
        X_closed = X + [X[0]] # close for easier looping
    
        # construct the lines that intersect the row we're working on
        # @TODO construct the lines before calling this (faster)
        lines = []
        bounds = []
        for i in range(len(Y)):
            y,yn = Y_closed[i],Y_closed[i+1]
            x,xn = X_closed[i],X_closed[i+1]
            if min(y,yn) <= y_val <= max(y,yn):
                # edge cases (@TODO handle this better)
                if yn == y:
                    # if we encounter a parallel line, just use it as the bounds
                    bounds = [x,xn]
                    break
                elif xn == x:
                    # if we encounter a purely vertical line, 
                    #   use that X as one of the bounds
                    bounds.append(x)
                else:
                    # a "normal" line encountered; construct it
                    m = (yn-y)/(xn-x)
                    b = y-m*x
                    lines.append([m,b])
    
        # the section (columns) that we want to replace are the intersection
        #   points of any intersecting lines:
        for m,b in lines:
            bounds.append(int((y_val-b)/m))
        # sort for convenience:
        bounds.sort()
        return bounds
   
