"""Custom AR detection.
"""
import cv2

# global variables (to be parameterized)
FIDUCIAL_CORNERS = 11

class ARDetector:
    """Object oriented approach to filtering and detecting AR
    tags in an individual image.
    """
    __debug = False
    def __init__(self, frame):
        self._original = frame.copy()
        self._frame = frame.copy()

    def detect(self):
        """Top level API for this class.

        Performs all operations (filtering, thresholding, etc.)
        Returns detected AR points and orientation.
        """
        # convert to grayscale
        self._frame = self.grayscale(self._frame)
        # filter
        self._frame = self.filter(self._frame)
        # threshold
        self._frame = self.threshold(self._frame)
        # perform detection
        return self._find_tags(self._frame)

    #------------------ STATIC API FUNCTIONS ------------------#

    @classmethod
    def grayscale(cls, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cls.__debug:
            cls.plot(gray, "gray")
        return gray

    @classmethod
    def filter(cls, frame):
        filtered = cv2.medianBlur(frame, 1)
        if cls.__debug:
            cls.plot(filtered, "filtered")
        return filtered

    @classmethod
    def threshold(cls, frame):
        ret,thresh = cv2.threshold(frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        if cls.__debug:
            cls.plot(thresh, "threshold")
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
        # get contours, hierarchy
        contours, hierarchy = cv2.findContours(frame.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

        if self.__debug:
            img = self._original.copy()
            cv2.drawContours(img,contours,-1,(0,255,0),3)
            self.plot(img, "all contours")
       
        # traverse contours
        #   we're looking for a child/parent pair that matches our internal
        #   shape and outside border

        # first find all bottommost children
        is_child = lambda h: h[2] == -1 and h[3] != -1
        children = [idx for idx, val in enumerate(hierarchy[0]) if is_child(val)]

        # sanity check that the number of points of each child is 
        #   more than 4 (our tag is a complex shape)
        # @TODO actually look for the number of vertices we 
        #   expect from the individual tag?
        
        children = [child for child in children if len(cv2.approxPolyDP(contours[child], 2, True)) > 4]

        # filter parents (note that this is conservative, we'll miss frames
        parents = [hierarchy[0][child][3] for child in children]
        parents = [parent for parent in parents if len(cv2.approxPolyDP(contours[parent],2,True)) == 4]

        # return the contours corresponding to the direct parents 
        #   of all our matching children (these should all be 
        #   precise rectangles)
        # @TODO actually calculate orientation!
        results = [(cv2.approxPolyDP(contours[parent],2,True),1) for parent in parents]

        if self.__debug:
            img = self._original.copy()
            cv2.drawContours(img,[contours[p] for p in parents],-1,(0,255,0),3)
            self.plot(img, "AR contours")

        return results



