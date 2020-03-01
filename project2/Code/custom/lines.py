"""Functions used in Line Fitting
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial import Polynomial

# PARAMETERS
POLY_DEGREES=2
MAX_RESIDUAL=50000

def histogram_y(thresh, verbose=False):
    """Calculate the histogram of a thesholded image (columnwise)
    """

    # histogram calculation
    hist = np.sum(thresh,0)//255 
    
    # define a cutoff (somewhat arbitrary) to separate 
    #  points into column buckets
    cutoff = np.mean(hist)
    bucket_indices = (hist>=cutoff).nonzero()[0]

    buckets = []
    start_idx = bucket_indices[0]
    prev_idx = bucket_indices[0]
    # actually separate into buckets
    for idx in bucket_indices[1:]:
        # check if it's the last bucket
        if idx == bucket_indices[-1]:
            buckets.append([start_idx, idx])
            break
        if idx - prev_idx == 1:
            prev_idx = idx
            continue
        # we've hit a rising edge
        if start_idx != prev_idx:
            buckets.append([start_idx, prev_idx])
        start_idx = idx
        prev_idx = idx

    if verbose:
        plot_histogram(hist, buckets)

    return buckets

def plot_histogram(hist, buckets):
    plt.plot(range(hist.size),hist)
    for b in buckets:
        x = b*2
        x.sort()
        y = [0, max(hist), max(hist),0]
        plt.plot(x,y,'-r')
    plt.show()


def polyfit(thresh, verbose=False, color=(0,0,255)):
    """Fit a polynomial line to each non-zero section of the given frame.
    """
    # initialize return image (with lines drawn)
    result = thresh.copy()
    result = cv2.cvtColor(result,cv2.COLOR_GRAY2RGB)

    # get histogram buckets
    buckets = histogram_y(thresh, verbose)

    # get the best fit poly approximation for each bucket
    good_lines = []
    for b in buckets:
        # get all frame points in the bucket
        subsection = thresh[:,b[0]:b[1]+1]

        # all the nonzero points (which we care about)
        points = np.where(subsection>0)
    
        # get polyline approx
        try:
            fit,res = Polynomial.fit(points[1]+b[0],points[0],POLY_DEGREES,full=True)
        except Exception:
            import pdb;pdb.set_trace()
        
        # ignore bad fits
        if res[0] > MAX_RESIDUAL:
            continue
   
        # if we've made it this far it's a good line
        good_lines.append(fit)

        # subsample line and plot
        pts = fit.linspace()
        for i,y in enumerate(pts[0]):
            y = int(y)
            x = int(pts[1][i])
            # print on image
            result[x,y] = np.array(color)

    return result, good_lines
    


