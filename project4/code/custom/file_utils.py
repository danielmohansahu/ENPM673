""" Functions related to file operations.

The scripts contained in this file are used for opening,
loading, saving, and other miscellaneous utilities
for file operation (specifically video and image files).
"""

import os
import sys
import time
from pathlib import Path
from collections import Generator
import numpy as np
import cv2

class VidWriter:
    def __init__(self, filename, fourcc, fps, size, isColor=True):
        self._writer = cv2.VideoWriter(filename, fourcc, fps, size, isColor)
    
    def __enter__(self):
        return self._writer

    def __exit__(self, type, value, traceback):
        self._writer.release()

class VidGenerator(Generator):
    """Return a generator for the given video file.

    This function returns an object that can be accessed sequentially
    (a la list comprehension, etc.) to get each video file.
    """
    def __init__(self, filepath, verbosity=1):
        # sanity check file and save metadata
        self._path = Path(filepath)
        if not self._path.is_file():
            raise RuntimeError("Unable to open video {}; no such file.".format(filepath))
        self.name = self._path.name
        self.path = self._path.as_posix()

        # open the video file
        self._video = cv2.VideoCapture(self.path)

        # get some metadata
        self.frame_count = int(self._video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fourcc = int(self._video.get(cv2.CAP_PROP_FOURCC))
        self.fps = int(self._video.get(cv2.CAP_PROP_FPS))
        self.size = (
            int(self._video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )

        # timing and debugging information
        self._verbosity = verbosity
        self._last_access_time = None
        self._start_access_time = time.time()
        self._current_frame = 0

    def send(self, ignored):
        """Core accessor of the underlying video file.
        """

        # timer / debugger
        #  note that this rests on the (shaky) assumption
        #  that the time *between* access is the processing
        #  time.
        if self._current_frame:
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*self._current_frame/self.frame_count), int(100*self._current_frame/self.frame_count)))
            sys.stdout.flush()
        
        self._current_frame += 1

        ret, frame = self._video.read()
        # check if we're done
        if frame is None:
            print("\nFinished processing in {}".format(time.time()-self._start_access_time))
            self.throw()
        return ret, frame

    def throw(self, type=None, value=None, traceback=None):
        self._video.release()
        raise StopIteration

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


