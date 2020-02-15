"""Custom AR detection.
"""
import cv2

class ARDetector:
    """Object oriented approach to filtering and detecting AR
    tags in an individual image.
    """
    debug = False
    def __init__(self, frame):
        self._original = frame
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
        if cls.debug:
            cls.plot(gray, "gray")
        return gray

    @classmethod
    def filter(cls, frame):
        filtered = cv2.medianBlur(frame, 25)
        if cls.debug:
            cls.plot(filtered, "filtered")
        return filtered

    @classmethod
    def threshold(cls, frame):
        thresh = cv2.adaptiveThreshold(
                frame,
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2)
        if cls.debug:
            cls.plot(thresh, "threshold")

    @classmethod
    def plot(cls, frame, text="frame"):
        # convenience function for debugging
        cv2.namedWindow(text, cv2.WINDOW_NORMAL)
        cv2.imshow(text, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cls.debug=False
        cv2.destroyAllWindows()

    #------------------ PRIVATE MEMBER FUNCTIONS --------------#

    def _find_tags(self, frame):
        # return a tuple of ([corners], orientation)
        return (None, None)



