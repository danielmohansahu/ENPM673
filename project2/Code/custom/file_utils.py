""" Functions related to file operations.

The scripts contained in this file are used for opening,
loading, saving, and other miscellaneous utilities
for file operation (specifically video and image files).
"""

import os
import sys
import time
import yaml
from pathlib import Path
from collections import Generator
import numpy as np
import cv2

def load_params(filename):
    """Load camera parameters from a given file."""
    with open(filename, 'r') as yamlfile:
        data = yaml.safe_load(yamlfile)
    K = np.array([float(val) for val in data["K"].split()])
    K.resize([3,3])
    D = np.array([float(val) for val in data["D"].split()])
    return K,D

def pics2video(image_files, output_file, fps=15):
    """Convert a given (ordered) container of image files to a video.
    """

    # get size and data from first image (assuming they're the same)
    frame = imread(image_files[0])
    size = (frame.shape[1], frame.shape[0])

    # create video write object
    video_writer = VidWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
   
    # status bar initialization
    count = 0
    total = len(image_files)

    with video_writer as writer:
        for image_file in image_files:
            frame = imread(image_file)
            writer.write(frame)

            # update status bar
            count += 1
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('='*int(20*count/total), int(100*count/total)))
            sys.stdout.flush()

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
        if self._verbosity:
            new_time = time.time()
            if self._current_frame:
                print("Frame #{}/{} processed in {:.3f}s ({:.3f}s total)".format(self._current_frame, self.frame_count, new_time-self._last_access_time, new_time-self._start_access_time))
            self._last_access_time = new_time
            self._current_frame += 1

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
