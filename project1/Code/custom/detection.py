"""Custom AR detection.
"""
import cv2
from .utils import Timer

# global variables (should be parameterized
MIN_SIDES_MATCH=7
MAX_SHAPE_MATCH=0.15
APPROX_POLY_ERROR=3

class ARDetector:
    """Object oriented approach to filtering and detecting AR
    tags in an individual image.
    """
    __debug = False
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
        with Timer("\tgrayscale",self.__debug):
            self._frame = self.grayscale(self._frame)
        if self.__debug:
            self.plot(self._frame, "gray")
        
        # filter
        with Timer("\tfilter",self.__debug):
            self._frame = self.filter(self._frame)
        if self.__debug:
            self.plot(self._frame, "filtered")
        
        # threshold
        with Timer("\tthreshold",self.__debug):
            self._frame = self.threshold(self._frame)
        if self.__debug:
            self.plot(self._frame, "threshold")
        
        # perform detection
        with Timer("\tdetect",self.__debug):
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
            cls.debug(False)
        cv2.destroyAllWindows()

    @classmethod
    def debug(cls, on_or_off):
        cls.__debug = on_or_off

    #------------------ PRIVATE MEMBER FUNCTIONS --------------#

    def _find_tags(self, frame):
        #@TODO use cv2.fitEllipse to get angle (must be a better way)
        #@TODO get AR ID (via location of inner contour??)  
        
        # get contours, hierarchy
        contours, hierarchy = cv2.findContours(frame.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

        if self.__debug:
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

        # return the rectangular approximation of the parent of each detected tag
        parents = [hierarchy[0][i][3] for i in tag_idxs]
        parents = [cv2.approxPolyDP(contours[p],APPROX_POLY_ERROR,True) for p in parents]
        
        # filter parents (note that this is conservative, we'll miss frames
        parents = [p for p in parents if len(p)==4]

        # append IDs (@TODO actually calculate this)
        results = [(p,1) for p in parents]

        if self.__debug:
            img = self._original.copy()
            cv2.drawContours(img, parents,-1,(0,255,0),3)
            self.plot(img, "AR contours")

        return results



