#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 00:17:16 2020

@author: Brenda
"""

import pdb 
import numpy as np
from numpy import linalg as linalg
import cv2
from matplotlib.pyplot import plot as plt



# Define some functions

#  Preprocess the image 
def preprocess_image(image):
    # preprocess
    grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(grey,3)
    ret, threshold = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
    return threshold

# Find the contours in the video frames
def find_contours(image):
    # find contours
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours - pick the 4 that define AR tgt
    #print('before', contours)
    image_contours = []  
    for j, cnt in zip(hierarchy[0], contours):

        epsilon = cv2.arcLength(cnt,True)
        cnt = cv2.approxPolyDP(cnt, 0.02*epsilon,True)
        if cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt) and len(cnt) == 4  :
            cnt=cnt.reshape(-1,2)
            if j[0] == -1 and j[1] == -1 and j[3] != -1:
                #print(cnt)
                image_contours.append(cnt)
                draw_contours = np.array(cnt, int)
    #print('after', image_contours)
    return image_contours, draw_contours

def AR_adjustment(contour):
    
    #print(contour)
    #print(np.shape(contour))
    pts = np.array(contour[0], int)
    print(pts)
    new_contour = []
    rect = np.zeros((4, 2), int)
    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    #print('rect', rect)
    
    new_contour = np.array(rect, int)

    return new_contour


    

#    
#    # Initialize the given constants
#    c1 = des[0,0]
#    c2 = des[0,1]
#    c3 = des[0,2]
#    c4 = des[0,3]
#    x1, x2, x3, x4 = 5, 150, 150, 5
#    y1, y2, y3, y4 = 5, 5, 150, 150
#
#    xp1, xp2, xp3, xp4 = 100, 200, 220, 100
#    yp1, yp2, yp3, yp4 = 100, 80, 80, 200
#
## Calculate A
#w1[0],w1[1],1,0,0,0,-c1[0]*w1[0],-c1[0]*w1[1],-c1[0]
#    A = np.array([
#         [-x1, -y1, -1,   0,   0,  0, x1*xp1, y1*xp1, xp1],
#         [  0,   0,  0, -x1, -y1, -1, x1*yp1, y1*yp1, yp1],
#         [-x2, -y2, -1,   0,   0,  0, x2*xp2, y2*xp2, xp2],
#         [  0,   0,  0, -x2, -y2, -1, x2*yp2, y2*yp2, yp2],
#         [-x3, -y3, -1,   0,   0,  0, x3*xp3, y3*xp3, xp3],
#         [  0,   0,  0, -x3, -y3, -1, x3*yp3, y3*yp3, yp3],
#         [-x4, -y4, -1,   0,   0,  0, x4*xp4, y4*xp4, xp4],
#         [  0,   0,  0, -x4, -y4, -1, x4*yp4, y4*yp4, yp4]], float)
#    
#    AT = np.transpose(A)
#    #print('A transpose is \n', AT)
#    
#    # Calculate the eigenvalues and eigenvectors of A*AT (9x9 matrix)
#    
#    M = A @ AT
#    w, v = linalg.eig(M)
#    
#    # Calculate the eigenvalues and eigenvectors of AT*A (8x8 matrix), which are
#    # also the right eigenvectors of A, vt
#    
#    MT = AT @ A
#    wt, V = linalg.eig(MT)
#            
#    # Compose S = diagonal matrix with singular values on diagonal
#    
#    I_arr = np.array([[wt[0], 0., 0., 0., 0., 0., 0., 0., 0.],
#           [0., wt[1], 0., 0., 0., 0., 0., 0., 0.],
#           [0., 0., wt[2], 0., 0., 0., 0., 0., 0.],
#           [0., 0., 0., wt[3], 0., 0., 0., 0., 0.],
#           [0., 0., 0., 0., wt[4], 0., 0., 0., 0.],
#           [0., 0., 0., 0., 0., wt[5], 0., 0., 0.],
#           [0., 0., 0., 0., 0., 0., wt[6], 0., 0.],
#           [0., 0., 0., 0., 0., 0., 0., wt[7], 0.]])
#    S = np.sqrt(I_arr)
#    print('The singular value matrix is an 8 x 9 matrix S = \n', S)
#    
#    # Transpose matrix vt to get VT, and check to see if multiplies to the 
#    # identity matrix
#    
#    VT = np.transpose(V)
#    #print('VT is the transpose of V, and is \n', VT)
#    print('Is vt@VT the 9 x 9 Identity matrix? yes, within rounding error\n', V@VT)
#    
#    # Use inverse of singular values ui= 1/sigma* A @ vi to find U
#    
#    U = np.empty([8,8],float)
#    
#    for i in range(7):
#        sigma = (1/np.sqrt(abs(w[i])))
#        U[:,i] = sigma* (A@V[:,i])
#    
#    #print('U is \n', U)
#    
#    # Check if U@np.transpose(U) = Identity matrix
#    
#    print('Is U@U_transpose the 8 x 8 Identity matrix? yes, within rounding error\n', V@VT)
#    
#    # Check to see if the SVD calculation equals A
#                    
#    print('Does U@S@VT = A? yes, to rounding error\n', (np.round(U@S@VT) ))
#    
#    # The solution to Ax= 0 is vt[:, 8], the last column vector of VT. It spans the 
#    # basis of the null space of A; Check to see it is in the nullspace
#    
#    Sol_vec = A@V[:,0]        
#    for i in range(8):
#        if Sol_vec[i] < 1e-7:
#            Sol_vec[i] = 0.
#            
#    print('The last column vector is in the nullspace of A; to within rounding error A@vt[;,8] = 0\n', A@V[:,8])
#    print('The solution to Ax = 0 is x = \n', V[:,8])
#
## Create the homography matrix
#
#    H = np.reshape(V[:,8], (3,3))
def homography(source,des):
    #print(des)
    #print(np.shape(des))
    
    c1 = des[0]
    c2 = des[1]
    c3 = des[2]
    c4 = des[3]
    
    

    w1 = source[0]
    w2 = source[1]
    w3 = source[2]
    w4 = source[3]

    A=np.array([[w1[0],w1[1],1,0,0,0,-c1[0]*w1[0],-c1[0]*w1[1],-c1[0]],
                [0,0,0,w1[0], w1[1],1,-c1[1]*w1[0],-c1[1]*w1[1],-c1[1]],
                [w2[0],w2[1],1,0,0,0,-c2[0]*w2[0],-c2[0]*w2[1],-c2[0]],
                [0,0,0,w2[0], w2[1],1,-c2[1]*w2[0],-c2[1]*w2[1],-c2[1]],
                [w3[0],w3[1],1,0,0,0,-c3[0]*w3[0],-c3[0]*w3[1],-c3[0]],
                [0,0,0,w3[0], w3[1],1,-c3[1]*w3[0],-c3[1]*w3[1],-c3[1]],
                [w4[0],w4[1],1,0,0,0,-c4[0]*w4[0],-c4[0]*w4[1],-c4[0]],
                [0,0,0,w4[0], w4[1],1,-c4[1]*w4[0],-c4[1]*w4[1],-c4[1]]])

    #Performing SVD
    U, s, VT = linalg.svd(A)

            # normalizing by last element of v
            #v =np.transpose(v_col)
    V = VT[8:,]/VT[8][8]

    H = np.reshape(V,(3,3))
    #print()
    return H


def superimpose_image(contour, image, source):
    print('des', np.shape(contour))
    cv2.imshow('image tag', image)
    cv2.waitKey(1)
    destination = contour  # destination points
    print(destination)
    source = np.array([[0,0],[511, 0],[511, 511],[0,511]],dtype=float) # points on source
    h = homography(source, destination)

    cv2.fillConvexPoly(image, destination.astype(np.int32), (255, 0, 255));
    
    #cv2.imshow('image tag', image)
    #cv2.waitKey(1)
    

    return image, h



def compensate_for_perspective(contour,image):

    dst = np.array([
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100]], float)
    #h, status = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)
    
    h, status = cv2.findHomography(contour, dst)

    warp = cv2.medianBlur(image.copy(),3)
    adjusted_image = cv2.resize(warp, dsize=None, fx=0.08, fy=0.08)
    
    return adjusted_image





def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(255,0,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(0,150,150),3)
        # draw top layer in red color
        img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

# @brief Calculates projection matrix
#
#  @param Homography
#
#  @return Projection matrix for 3D transformation
#
def projection_matrix(homography):
   # K is the transpose of the camera matrix (given)
   K =np.array([[1406.08415449821,                 0,0],
                [ 2.20679787308599, 1417.99930662800,0],
                [ 1014.13643417416, 566.347754321696,1]])
   B_tilda = np.dot(linalg.inv(K), homography)
   col_1 = B_tilda[:, 0]
   col_2 = B_tilda[:, 1]
   col_3 = B_tilda[:, 2]
   lambda1 = 1/(2*(linalg.norm(col_1, 2) * linalg.norm(col_2, 2)))
   r_1 = col_1 * lambda1 
   r_2 = col_2 * lambda1 
   translation = col_3 * lambda1 
   c = r_1 + r_2
   p = np.cross(r_1, r_2)
   d = np.cross(c, p)
   r_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / np.sqrt(2))
   r_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / np.sqrt(2))
   r_3 = np.cross(r_1, r_2)

   projection = np.stack((r_1, r_2, r_3, translation)).T
   print(projection)
   return np.dot(K, projection)


def Cube3D(projection_matrix,image):
    # how to render a cube
    #V = np.zeros(8, [("a_position", np.float32, 3)])
    #V["a_position"] = [[ 1, 1, 1], [-1, 1, 1], [-1,-1, 1], [ 1,-1, 1],
    #              [ 1,-1,-1], [ 1, 1,-1], [-1, 1,-1], [-1,-1,-1]]
    axis = np.float32([[0,0,0,1],[0,512,0,1],[512,512,0,1],[512,0,0,1],[0,0,-512,1],[0,512,-512,1],[512,512,-512,1],[512,0,-512,1]])
    Proj = np.matmul(axis,projection_matrix.T)
    
    print(Proj)
    # Normalize the matrix
    Norm1 = np.divide(Proj[0],Proj[0][2])
    Norm2 = np.divide(Proj[1],Proj[1][2])
    Norm3 = np.divide(Proj[2],Proj[2][2])
    Norm4 = np.divide(Proj[3],Proj[3][2])
    Norm5 = np.divide(Proj[4],Proj[4][2])
    Norm6 = np.divide(Proj[5],Proj[5][2])
    Norm7 = np.divide(Proj[6],Proj[6][2])
    Norm8 = np.divide(Proj[7],Proj[7][2])

    points = np.vstack((Norm1,Norm2,Norm3,Norm4,Norm5,Norm6,Norm7,Norm8))
    final_2d=np.delete(points,2, axis=1)
    draw(image,final_2d)
    return image
'''

def projectPoints_undist(points3d, R, K, T):
        """
            projects 3d points into 2d ones with
            no distortion
        :param points3d: {n x 3}
        :return:
        """
        pts2d, _ = cv2.projectPoints(points3d,
                                     np.dot(R, points3d),
                                     np.dot(T,points3d),
                                     np.dot(K, points3d), 0)
        pts2d = np.squeeze(pts2d)
        if len(pts2d.shape) == 1:
            pts2d = np.expand_dims(pts2d, axis=0)
        return pts2d 
    

def draw(image, contours, R, T, K):
    
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
    print()
    imgpts = projectPoints_undist(axis, R,K,T )
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    image = cv2.drawContours(image, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        image = cv2.line(image, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    image = cv2.drawContours(image, [imgpts[4:]],-1,(0,0,255),3)
    cv2.imshow('cube', image)
    cv2.waitKey(1)
    return image
    '''


def image_processor(cap,source):
    count = 0
    image_array=[]
    while (1):
        ret, image = cap.read()
        if ret == True:
            height, width, c = image.shape
            size = (width,height)
            image_preprocessed = preprocess_image(image.copy())
 
            #cv2.imshow('contours', image_preprocessed)
            #cv2.waitKey(1)
            contours, draw_contours = find_contours(image_preprocessed)

            #cv2.imshow('contours', image)
            #cv2.waitKey(1)
            #print(corners)
            if(len(contours)==0):   # sometimes we don't get a contour, so use last one
                contours = previous_contours
            #print(contours)
            
            
            print(np.shape(contours))
            #c = np.array( [ [1121,  524],[1164,  586],[1095,  616],[1053,  553]], int)
            #image = cv2.drawContours(image, contours[1], 0, (0,255,0),4)
            image = cv2.drawContours(image, contours,   0, (0,255,0),4)
            #cv2.imshow('contours', image)
            #cv2.waitKey(1)
        
            new_contours = AR_adjustment(contours)
            cv2.imshow('new_contours', image)
            cv2.waitKey(1)
            adjusted_contours = compensate_for_perspective(new_contours, image_preprocessed)
            
            
            image_super, h = superimpose_image(adjusted_contours, image, source)   ##superimposes a black 'square' over AR tgt
            #cv2.imshow('super', image)
            #cv2.waitKey(1)
            projection_matrix = projection_matrix(h)
            image_cube = Cube(projection_matrix, image)  # semblance of a cube
            #cv2.imshow('new_contours', image)
            #cv2.waitKey(1)
            previous_contours = contours
            count += 1
    
            cv2.imwrite("face-{:03d}.jpg".format(count+1),image)
            image_array.append(image)
            #cv2.imshow('frame',image)
            #cv2.waitKey(0)
            
    return image_array,size


  
###############################################################################
######  Main code - reads the image, video, and writes processed image   ######
######  to an *.avi file.                                                ######
###############################################################################


source=cv2.imread('lena.png')
cap = cv2.VideoCapture('Tag0.mp4')
   
image_array,size = image_processor(cap, source)
'''
    out=cv2.VideoWriter('Cube_Tracking.avi',cv2.VideoWriter_fourcc(*'DIVX'),20 ,size)
    for i in range(len(image_array)):
        out.write(image_array[i])
    out.release()
'''    
    
