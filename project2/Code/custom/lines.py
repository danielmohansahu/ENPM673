"""Functions used in Line Fitting
"""

import math
import itertools
import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial import Polynomial

# PARAMETERS
MIN_CONTOUR_AREA=50
MAX_CONTOUR_AREA=10000
BUCKET_OVERLAP=20
MAX_RES_ERROR=500000
POLY_DEGREES=1
MAX_NUMBER_OUTLIERS=3

def polyfit(thresh, verbose=False, color=(0,0,255)):
    """Fit a polynomial line to each non-zero section of the given frame.
    """

    # initialize return image (with lines drawn)
    result = thresh.copy()
    result = cv2.cvtColor(result,cv2.COLOR_GRAY2RGB)

    # get all contours:
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # filter contours and separate into "possible lane" buckets
    contour_buckets = filter_contours(contours, [0,thresh.shape[1]])

    # get the best fit poly approximation for each contour
    good_lines = []
    good_weights = []
    good_error = []
    for bucket in contour_buckets:
        # find best fit (with outlier rejection)
        fits,weights,error = ransac_fit(bucket)

        # if we've made it this far it's a good line
        good_lines += fits
        good_weights += weights
        good_error += error

    lanes = filter_lines(good_lines, good_weights, good_error)

    # subsample line and plot
    for fit in lanes:
        pts = fit.linspace()
        for i,y in enumerate(pts[0]):
            y = int(y)
            x = int(pts[1][i])
            # print on image (if within bounds)
            if x < result.shape[0] and y < result.shape[1]:
                result[x,y] = np.array(color)

    return result, good_lines

def ransac_fit(bucket):
    """Find the best matching polyfit to the given contours (with outlier rejection)
    """
    good_fits = []
    weights = []
    error = []
    for i in range(min(len(bucket),MAX_NUMBER_OUTLIERS)):
        for combination in itertools.combinations(bucket,len(bucket)-i):
            # skip empty combinations
            if len(combination)==0:
                continue
            # put all points in this combination together
            x_points = np.array([])
            y_points = np.array([])
            for contour in combination:
                x_points = np.concatenate((x_points,contour[:,0][:,0]))
                y_points = np.concatenate((y_points,contour[:,0][:,1]))
    
            # ignore points and vertical lines:
            if len(set(x_points)) == 1:
                continue
            
            # get polyline approx
            fit,res = Polynomial.fit(x_points,y_points,POLY_DEGREES,full=True)
            
            # check if this is an ok fit:
            if res[0][0] < MAX_RES_ERROR:
                good_fits.append(fit)
                error.append(res[0][0])
                weights.append(len(x_points))
    return good_fits, weights, error
    

def filter_lines(lines, weights, error):
    """ Apply known heuristics to our detected lines. 
    
    Tries to return the two best lane estimations.

    Assumptions:
     - Lines should be close to vertical.
     - Lanes should be roughly parallel.
    """

    # cost function we're trying to minimize
    cost = lambda angle,weight,res: 0.33*angle + 0.33*weight + 0.33*res

    # normalize our parameters
    error_scale = max(error)
    weight_scale = max(weights)
    angle_scale = math.pi/2.0

    # filter our lines with horizontal slopes
    costs = []
    for i in range(len(lines)):
        # get line angle (from vertical)
        angle = math.atan(abs(lines[i].coef[1]))

        # scale everything and calculate cost
        a = angle/angle_scale           
        w = weight_scale/weights[i]     # inverted (higher weight is better)
        e = error[i]/error_scale
        costs.append(cost(a,w,e))
   
    # get two "Best" fits:
    sorted_costs = costs[:]
    sorted_costs.sort()
    
    best_fit = []
    best_fit.append(lines[costs.index(sorted_costs[0])])
    best_fit.append(lines[costs.index(sorted_costs[1])])

    # @TODO 

    return best_fit

def filter_contours(contours, xbounds):
    """ Apply known heuristics to our detected lane contours. 
    
    Assumptions:
     - Each lane is fully in view (i.e. no sharp turns)
     - We can filter our really small and really big contours  
     - Lanes don't have overlapping X points (e.g. they can be 
        separate via histogram analysis in Y
    
    Returns:
     - Contours separated into buckets, each of which could be a lane.
    """
    filtered_contours = []
    for contour in contours:
        # filter out too big / small contours
        area = cv2.contourArea(contour)
        if MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA:
            # check if any contour elements are on our X boundaries 
            if all([x not in xbounds for x in contour[:,0][:,0]]):
                filtered_contours.append(contour)
    
    # separate into buckets
    contour_buckets = histogram_sort(filtered_contours)
    return contour_buckets

def histogram_sort(contours, verbose=True):
    """Separate the given set of contours into buckets based on X position.

    #@TODO make this a lot more efficient
    """
    def merge_buckets(*args):
        min_ = min([a[0] for a in [arg for arg in args][0]])
        max_ = max([a[1] for a in [arg for arg in args][0]])
        return [min_, max_]

    bucket_bounds = []
    for contour in contours:
        # get contour bounds
        bounds = [min(contour[:,0][:,0]),max(contour[:,0][:,0])]

        # check if this overlaps with any other buckets
        matching_buckets = []
        for b in bucket_bounds:
            if (b[0]-BUCKET_OVERLAP<=bounds[0]<=b[1]+BUCKET_OVERLAP) or (b[0]-BUCKET_OVERLAP<=bounds[1]<=b[1]+BUCKET_OVERLAP):
                matching_buckets.append(b)

        # check if this is a new bucket or we need to merge 
        if len(matching_buckets) == 0:
            bucket_bounds.append(bounds)
        else:
            # remove matching buckets
            for b in matching_buckets:
                bucket_bounds.remove(b)
            
            # add our current bounds (to be merged)
            matching_buckets.append(bounds)
            
            # add back in merged versions
            bucket_bounds.append(merge_buckets(matching_buckets))

    # now that we've got our main buckets, separate our contours into them
    buckets = [[] for _ in range(len(bucket_bounds))]
    for contour in contours:
        # take a random point in our bucket
        point_x = contour[0][0][0]
        
        # find the bucket we belong in
        for i,bounds in enumerate(bucket_bounds):
            if bounds[0]<=point_x<=bounds[1]:
                buckets[i].append(contour)
                break
    
    # further segment buckets to separate contours that coincide in rows
    segmented_buckets = []
    for bucket in buckets:
        segmented_buckets.append(recurse_bucket_overlap(bucket))

    return segmented_buckets


def recurse_bucket_overlap(bucket):
    """Separate a given bucket of contours into all non-overlapping (in X) subbuckets
    """
    # initialize recursion variables
    remainder=bucket[:]
    subbuckets = []
    current_bucket = []
    
    # call subfunction
    _recurse_bucket_overlap(remainder, current_bucket, subbuckets)
    return subbuckets
    

def _recurse_bucket_overlap(remainder, current_bucket, subbuckets):
    # internal method for recursively finding all subbuckets
    
    # traverse this current path
    while len(remainder) != 0:
        # get next contour
        contour = remainder.pop()

        # get contour bounds
        bounds = [min(contour[:,0][:,0]),max(contour[:,0][:,1])]

        # check if there's any overlap
        overlapping_contours = []
        for rc in remainder:
            # check remainder contour bounds
            rc_bounds = [min(contour[:,0][:,0]),max(contour[:,0][:,1])]
            if (rc_bounds[0]<=bounds[0]<=rc_bounds[1]) or (rc_bounds[0]<=bounds[1]<=rc_bounds[1]):
                # we've got overlap; separate this out
                overlapping_contours.append(rc)

        # if we've got no overlap, just continue
        if len(overlapping_contours)==0:
            current_bucket.append(contour)
        else:
            # remove overlapping contours from remainder
            for overlap in overlapping_contours:
                remainder.remove(overlap)

            # append current contour
            overlapping_contours.append(contour)

            # recurse down into each of these branching paths with our remainder
            for overlap in overlapping_contours:
                _recurse_bucket_overlap(remainder+[overlap], current_bucket, subbuckets)

        # check if we're done with this path
        if len(remainder)==0:
            subbuckets += current_bucket + [contour]






