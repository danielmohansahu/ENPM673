"""Classes and scripts related to Homography and Tracking.
"""

import numpy as np
import cv2

def warp_image(frame, H, output_shape):
    """Warp the given image with the given homography matrix.
    """
    dst = np.zeros(output_shape)

    # construct homogeneous form of indices
    size_x, size_y = frame.shape[:-1]
    dst_hom = np.array([
        np.array([[i]*size_x for i in range(size_y)]).flatten(),
        [*range(size_x)]*size_y,
        [1]*size_x*size_y
        ])

    # homographize and normalize:
    homography = H@dst_hom
    x_f = homography[0,:]/homography[2,:]
    y_f = homography[1,:]/homography[2,:]

    for i in range(len(x_f)):
        dst[int(y_f[i]), int(x_f[i]),:] = frame[dst_hom[1,i],dst_hom[0,i],:]

    return dst

def get_corners(image):
    """Get the outer corners of the given image.
    """
    y,x,_ = image.shape
    corners = np.array([[[0,0]],[[x,0]],[[x,y]],[[0,y]]])
    return corners

def to_homogeneous(points):
    """Convert a given list of np arrays to Homogeneous coordinates.
    """
    homogeneous = []
    for point in points:
        homogeneous.append(point.flatten().tolist() + [1])
    return np.array(homogeneous)

def replace_section(src, replacement, contour):
    """Replace the contour in src with the same contour in dst.
    Images must be the same size.
    """
    if src.shape != replacement.shape:
        raise RuntimeError("Unable to process images of different shapes. {} vs {}".format(src.shape, replacement.shape))

    # initialize result image
    result = src.copy()

    # initialize some loop variables
    size_y, size_x = src.shape[:-1]
    X_c = [c[0][0] for c in contour]    # contour X value list
    Y_c = [c[0][1] for c in contour]    # contour Y value list

    for y_pix in range(size_y):
        # find out which columns of the given row should be replaced, and replace.
        bounds = get_contour_x_bounds(y_pix,X_c,Y_c)
        if bounds:
            result[y_pix,bounds[0]:bounds[1]] = replacement[y_pix,bounds[0]:bounds[1]]

    return result

def get_contour_x_bounds(y_val, X, Y):
    """Calculate the minimum and maximum X that are still within
    the given contour (X,Y) for the given Y value.

    @TODO handle edge cases (on a corner, on a line) better
    """
    # return empty range if we're above or below the contour
    if y_val < min(Y) or y_val > max(Y):
        return None
    
    # general case; find the two lines our row intersects
    Y_closed = Y + [Y[0]] # close for easier looping
    X_closed = X + [X[0]] # close for easier looping

    # construct the lines that intersect the row we're working on
    # @TODO construct the lines before calling this (faster)

    lines = []
    for i in range(len(Y)):
        y,yn = Y_closed[i],Y_closed[i+1]
        x,xn = X_closed[i],X_closed[i+1]
        if min(y,yn) <= y_val <= max(y,yn):
            m = (yn-y)/(xn-x)
            b = y-m*x
            lines.append([m,b])

            # edge case (@TODO handle this better)
            if m == 0:
                bounds = [x,xn]
                bounds.sort
                return bounds

    # the section (columns) that we want to replace are the intersection
    #   points of our intersecting lines:
    bounds = []
    for m,b in lines:
        bounds.append(int((y_val-b)/m))
    # sort for convenience:
    bounds.sort()
    return bounds

def get_homography(template_corners, corners, orientation):
    """Compute homography between sets of corners.
    """
    # sanity checks
    if len(corners) != len(template_corners):
        raise RuntimeError("Given different sized corner arrays for homography; {} vs {}".format(template_corners,corners))
    
    # reorder corners
    #  @TODO do this differently and elsewhere!
    corners = [corners[(i+orientation)%4] for i in range(4)]

    # compute set of homography linear equations:
    A = np.zeros([2*len(corners), 9])
    for i in range(len(corners)):
        x1w,y1w = corners[i].flatten().tolist()
        x1c,y1c = template_corners[i].flatten().tolist()
        A[2*i]   = np.array([x1w,y1w,1,0,0,0,-x1c*x1w,-x1c*y1w,-x1c])
        A[2*i+1] = np.array([0,0,0,x1w,y1w,1,-y1c*x1w,-y1c*y1w,-y1c])

    # calculate SVD
    U,D,V = np.linalg.svd(A)

    # select smallest eigenvalue and normalize
    # note that we're selecting a row, since SVD returns the 
    #   transpose of V
    H = V[-1,:]/V[-1,-1]
    H.resize([3,3])

    return H


