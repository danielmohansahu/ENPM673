""" Functions related to file operations.

The scripts contained in this file are used for opening,
loading, saving, and other miscellaneous utilities
for file operation (specifically video and image files).
"""

import os
from pathlib import Path
from collections import Generator
import numpy as np
import cv2

def vidshow(filepath):
    """ Display the given video using our custom generator.
    """
    video = VidGenerator(filepath)

    for _,frame in video:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow("frame",gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

class VidWriter:
    def __init__(self, filename, fps, fourcc, size):
        self._writer = cv2.VideoWriter(filename, fps, fourcc, size)
    
    def __enter__(self):
        return self._writer

    def __exit__(self, type, value, traceback):
        self._writer.release()

class VidGenerator(Generator):
    """Return a generator for the given video file.

    This function returns an object that can be accessed sequentially
    (a la list comprehension, etc.) to get each video file.
    """
    def __init__(self, filepath):
        # sanity check file and save metadata
        self._path = Path(filepath)
        if not self._path.is_file():
            raise RuntimeError("Unable to open video {}; no such file.")
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

    def send(self, ignored):
        ret, frame = self._video.read()
        # check if we're done
        if frame is None:
            self.throw()
        return ret, frame

    def throw(self, type=None, value=None, traceback=None):
        self._video.release()
        raise StopIteration

def imread(filepath, color=cv2.IMREAD_UNCHANGED):
    """Attempt to load a given image file.

    This is just a wrapper around cv2 imread, but we actually
    check if the file exists.
    """
    if not os.path.isfile(filepath):
        raise RuntimeError("Cannot load {}; file does not exist.".format(filepath))

    # actually load the file
    img = cv2.imread(filepath, color)

    if img is None:
        raise RuntimeError("Failed to load {}; is it an image file?".format(filepath))
    return img
