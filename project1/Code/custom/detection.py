"""Custom AR detection.
"""
import numpy as np
import cv2
from .utils import Timer

# global variables (should be parameterized
MIN_SIDES_MATCH=7
MAX_SHAPE_MATCH=0.5
APPROX_POLY_ERROR=3
INDEX_TO_ID = lambda idx: 2**idx

class ARDetector:
    """Object oriented approach to filtering and detecting AR
    tags in an individual image.
    """
    __verbosity = 0
    def __init__(self, frame, reference_tag):
        self._original = frame.copy()
        self._frame = frame.copy()

        # extract contours from the given reference image
        cnts, hier = cv2.findContours(reference_tag, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) != 1:
            raise RuntimeError("Unexpected number of contours extracted from given reference tag image.")
        self._reference_contour = cnts[0]

    def detect(self):
        """Top level API for this class.

        Performs all operations (filtering, thresholding, etc.)
        Returns detected AR points and orientation.
        """
        # convert to grayscale
        with Timer("\tgrayscale",self.__verbosity):
            self._frame = self.grayscale(self._frame)
        if self.__verbosity > 2:
            self.plot(self._frame, "gray")
        
        # filter
        with Timer("\tfilter",self.__verbosity):
            self._frame = self.filter(self._frame)
        if self.__verbosity > 2:
            self.plot(self._frame, "filtered")
        
        # threshold
        with Timer("\tthreshold",self.__verbosity):
            self._frame = self.threshold(self._frame)
        if self.__verbosity > 2:
            self.plot(self._frame, "threshold")
        
        # perform detection
        with Timer("\tdetect",self.__verbosity):
            detects = self._find_tags(self._frame)

        return detects

    #------------------ STATIC API FUNCTIONS ------------------#

    @staticmethod
    def grayscale(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray

    @staticmethod
    def filter(frame):
        filtered = cv2.medianBlur(frame, 1)
        return filtered

    @staticmethod
    def threshold(frame):
        ret,thresh = cv2.threshold(frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return thresh

    @classmethod
    def plot(cls, frame, text="frame"):
        # convenience function for debugging
        cv2.namedWindow(text, cv2.WINDOW_NORMAL)
        cv2.imshow(text, frame)

        key = cv2.waitKey(0)
        if key == ord('q'):
            # stop debugging on 'q'
            cls.debug(0)
        cv2.destroyAllWindows()

    @classmethod
    def debug(cls, verbosity):
        cls.__verbosity = verbosity

    #------------------ PRIVATE MEMBER FUNCTIONS --------------#

    def _find_tags(self, frame):
        #@TODO use cv2.fitEllipse to get angle (must be a better way)
        #@TODO get AR ID (via location of inner contour??)  
        
        # get contours, hierarchy [next, prev, child, parent]
        contours, hierarchy = cv2.findContours(frame.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

        if self.__verbosity > 2:
            img = self._original.copy()
            cv2.drawContours(img,contours,-1,(0,255,0),3)
            self.plot(img, "all contours")
      
        # get all contour matches (based on approxPolyDP points and matchShapes):
        tag_idxs = []
        for i,cnt in enumerate(contours):
            # get rough number of sides:
            sides = len(cv2.approxPolyDP(cnt, APPROX_POLY_ERROR, True))
            # get comparison to template
            shape_match = cv2.matchShapes(self._reference_contour, cnt, cv2.CONTOURS_MATCH_I1, 0)
            if sides > MIN_SIDES_MATCH and shape_match < MAX_SHAPE_MATCH:
                # consider this a tag
                tag_idxs.append(i)

        # get the parent index of each detected tag
        parents = [hierarchy[0][i][3] for i in tag_idxs]
        
        # determine the orientation of the parent
        #   the "first" point on our contour should be the closest
        #   point to the child contour
        corners = []
        ids = []
        for i, parent in enumerate(parents):
            tag = tag_idxs[i]

            # simplify parent and child contours
            parent_contour = cv2.approxPolyDP(contours[parent], 
                                              APPROX_POLY_ERROR, 
                                              True)
            tag_contour = cv2.approxPolyDP(contours[tag], 
                                           APPROX_POLY_ERROR, 
                                           True)

            # skip parent contours that aren't roughly square
            if len(parent_contour) != 4:
                continue

            # align contours
            aligned_contour = self._align_closest_point(tag_contour, parent_contour)
            corners.append(aligned_contour)

            # get fiducial ID (based on inner contour)
            tag_child_idx = hierarchy[0][tag][2]
            tag_child_contour = None if tag_child_idx==-1 else contours[tag_child_idx]
            ids.append(self._tag_id(aligned_contour, tag_child_contour))
        
        if self.__verbosity > 2:
            img = self._original.copy()
            cv2.drawContours(img, corners, -1, (0,255,0), 3)
            self.plot(img, "AR contours")

        return (corners, ids)

    def _align_closest_point(self, ref, src, offset=2):
        """ Orient the given contours.

        This function returns a reordered src such that its first 
        point is the closest point to ref[0] plus the given offset.
        """
        # get the indices of the closest points on ref/src
        _,src_idx = self._get_closest_points(ref, src)

        # return a reordered src copy
        idx = (src_idx + offset) % src.shape[0]
        result = np.concatenate((src[idx:],src[:idx]))
        return result

    def _tag_id(self, tag_contour, child_contour):
        """ Extract the ID of the tag.

        Our assumption is that the inner contour represents
        the bitwise ID of the tag, and that only 1 bit will
        be encoded.
        
        Note that the given videos don't seem to follow this
        naming convention properly
        """
        # no inner contour is tag ID #0
        if child_contour is None:
            return 0

        # determine where our inner contour is relative to our tag
        tag_idx,_ = self._get_closest_points(tag_contour, child_contour)

        return INDEX_TO_ID(tag_idx)

    def _get_closest_points(self, c1, c2):
        """Returns the indices of the two closest points in each contour.
        """
        min_dist = np.inf
        c1_idx = -1
        c2_idx = -1
        for i, c1_point in enumerate(c1):
            for j, c2_point in enumerate(c2):
                dist = np.linalg.norm(c1_point-c2_point)
                if dist < min_dist:
                    c1_idx = i
                    c2_idx = j
                    min_dist = dist
        return (c1_idx,c2_idx)


