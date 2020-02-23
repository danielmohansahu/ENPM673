#!/usr/bin/env python3
import os
import time
import code
import argparse
import numpy as np
from numpy import linalg as linalg
import cv2

from custom import file_utils
from custom.tracking import ARTracker
from custom.detection import ARDetector

TEST_FILE = "Tag0.mp4"
AR_FILE = "ref_marker.png"
TEMPLATE_FILE = "Lena.png"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=TEST_FILE, help="Input video containing fiducial information.")
    parser.add_argument("--verbosity", type=int, default=1, help="Set verbosity level (0 is none, 1 is console output, 2 is images).")
    return parser.parse_args()


#  Added functions for tracking and drawing cube 

'''
#  if we end up doing a camera calibration to get the distortion matrix 
#  (separate from camera matrix, K)), this may be helpful
def projectPoints_undist(points3d, R, K, T, distortion_Matrix):
        """
            projects 3d points into 2d ones with
            no distortion
        :param points3d: {n x 3}
        :return:
        """
        pts2d, _ = cv2.projectPoints(points3d,
                                     np.dot(R, points3d),
                                     np.dot(T,points3d),
                                     np.dot(K, points3d), distortion_Matrix)
        pts2d = np.squeeze(pts2d)
        if len(pts2d.shape) == 1:
            pts2d = np.expand_dims(pts2d, axis=0)
        return pts2d 

    '''    

# To build the cube and display it
def cube(proj_mat,image):
    
    # Convert to homogeneous coordinates
    axis = np.float32([[0,0,0,1],[0,512,0,1],[512,512,0,1],[512,0,0,1],
                       [0,0,-512,1],[0,512,-512,1],[512,512,-512,1],[512,0,-512,1]])
    Proj= np.matmul(axis,np.transpose(proj_mat))
    
    # Normalize the matrix by dividing by r3, first points are bottom of cube
    P1 = np.divide(Proj[0],Proj[0][2])
    P2 = np.divide(Proj[1],Proj[1][2])
    P3 = np.divide(Proj[2],Proj[2][2])
    P4 = np.divide(Proj[3],Proj[3][2])
    # Last points are top of cube
    P5 = np.divide(Proj[4],Proj[4][2])
    P6 = np.divide(Proj[5],Proj[5][2])
    P7 = np.divide(Proj[6],Proj[6][2])
    P8 = np.divide(Proj[7],Proj[7][2])
    
    points = np.vstack((P1,P2,P3,P4,P5,P6,P7,P8))
    imgpts=np.delete(points,2, axis=1)
    img = draw(image, imgpts)
    return img 


# To draw the cube at each frame
def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in blue
    img = cv2.drawContours(img, [imgpts[:4]],-1,(255,0,0) ,3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (50, 200, 0), 3)
        # draw top layer in red color
        img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

# To calculate the projection matrix to transform the contour points to the cube    
def projection_matrix(homography):
   # K is the camera matrix (given)
   K =np.array([[1406.08415449821,                 0,0],
                [ 2.20679787308599, 1417.99930662800,0],
                [ 1014.13643417416, 566.347754321696,1]])
   K = np.transpose(K)
   B_tilda = np.dot(linalg.inv(K), homography)
   #Check sign of B_tilda
   if (linalg.det(B_tilda)) < 0:
       B_tilda *= -1 
   print('B',B_tilda)    
   col_1 = B_tilda[:, 0]
   col_2 = B_tilda[:, 1]
   col_3 = B_tilda[:, 2]
   #lam = ((np.linalg.norm(np.matmul(Kinv,H_cube[:,0]))+(np.linalg.norm(np.matmul(Kinv,H_cube[:,1]))))/2)
   # l = math.sqrt(la.norm(col_1, 2) * la.norm(col_2, 2))
   lambda1 = 1/((linalg.norm(col_1, 2) + linalg.norm(col_2, 2))/2)
   #rint('l',lambda1)
   r_1 = col_1 * lambda1 
   r_2 = col_2 * lambda1
   r_3 = np.cross(r_1, r_2)
   translation = col_3 * lambda1 
   projection = np.stack((r_1, r_2, r_3, translation)).T
   print('Projection Matrix', projection)
   return np.dot(K, projection)



if __name__ == "__main__":
    # parse command line args
    args = parse_args()
    
    # get template file (lena) and reference tag file
    template = file_utils.imread(TEMPLATE_FILE)
    reference_tag = file_utils.imread(AR_FILE,0)

    # initialize tracker and set class debugging
    tracker = ARTracker(template)
    ARDetector.debug(args.verbosity)
    ARTracker.debug(args.verbosity)

    # initialize our video IO
    vidgen = file_utils.VidGenerator(args.video)
    output_file = "processed_" + os.path.basename(args.video)
    video_writer = file_utils.VidWriter(output_file, cv2.VideoWriter_fourcc(*'DIVX'), vidgen.fps, vidgen.size)
    
    # process every frame in the given video
    process_start = time.time()
    frame_count = 0
    with video_writer as writer:
        for ret,frame in vidgen:
            # metadata
            frame_start = time.time()
            frame_count += 1
            # Create a black image
            #black = np.zeros((1980, 1080), np.uint8)
            detector = ARDetector(frame, reference_tag)
            detections, ids = detector.detect()

            for corners in detections:
                frame, homography = tracker.track(frame, corners)
'''
question here - I put this in because the code broke sometimes, same as below
'''    
            if detections != []:
                print('contours', detections[0])
            # Create the cube
            
            s = 512
            source = np.array([[0,0], [s,0], [s,s],[0,s]])
      
            
            # Maybe some frames aren't getting a detection?? Stops program when []
            '''
question here - I put this in because the code broke sometimes
'''     
            pts = np.array(detections)
            if pts != []:
                pts_im = pts.reshape(4,2)
               
            
            H = ARTracker.get_homography( pts_im, source)
            proj_mat = projection_matrix(H)
            
            image = cube(proj_mat, frame)
            cv2.imshow('cube', image)
            cv2.waitKey(1)
           
           
            writer.write(frame)
            if args.verbosity:
                ctime = time.time()
                #print("Found ids {} in frame #{}/{} in {:.3f}s ({:.3f}s total".format(ids, frame_count, vidgen.frame_count, ctime-frame_start, ctime-process_start))


