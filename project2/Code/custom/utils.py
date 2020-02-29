"""Collection of miscellaneous useful functions.
"""
import time

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
