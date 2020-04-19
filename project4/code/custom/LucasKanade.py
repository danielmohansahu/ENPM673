"""Python implementation of the Lucas-Kanade algorithm for template tracking via Affine Transformation.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2

# Notes/Improvements:
#  - implement the Huber Loss M-estimator mentioned in the problem statement.
#  - implement max frame to frame transforms (rotation and translation) to prevent 
#     a single bad track from throwing off the entire sequence
#  - investigate switching entirely to either cv2 or np; this column vs. row order
#     switching is a huge headache.

class LucasKanade:

    def __init__(self, template, bounding_box):
        """Initialize the Lucas-Kanade algorithm.

        Args:
            template:   Image containing the template to track.
            bbox:       Bounding box of the template.
        """

        # remove everything from the template image except the bounding box
        self.shape = tuple(template.shape[1::-1])

        # compute homography from template image to ROI
        bb = bounding_box
        tpts = np.array([[bb[0],bb[1]],[bb[0]+bb[2],bb[1]],[bb[0]+bb[2],bb[1]+bb[3]],[bb[0],bb[1]+bb[3]]])
        ipts = np.array([[0,0],[self.shape[0],0],[self.shape[0],self.shape[1]],[0,self.shape[1]]])

        # get homography; note that this is itself just an affine transform
        self.H = cv2.findHomography(tpts,ipts)[0]

        # get our template
        self.template = cv2.warpPerspective(template, self.H, dsize=self.shape)
        self.template = np.float32(self.template)

        # initialize our parameter estimate (to zero)
        self.p = np.zeros((2,3),dtype=np.float32)
        self.p_hist = [] 

        # initialize certain constant parameters
        self.J = np.zeros((self.shape[1],self.shape[0],2,6))
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                self.J[y,x] = np.array([
                    [x,0,y,0,1,0],
                    [0,x,0,y,0,1]
                ])

        # other variables
        self.epsilon = 0.01     # stop criterion; min norm of affine delta to finish
        self.max_count = 1000   # maximum allowed number of iterations
        self.sigma = 0.8        # sigma for Huber Loss
        self.avg_frames = 3     # number of frames to use in moving average

    def average(self, p, save=True):
        """Return the moving average estimate of the current transform.
        """
        # update history of points
        new_p_hist = self.p_hist + [p]
        if len(new_p_hist) > self.avg_frames:
            new_p_hist.pop(0)

        # calculate average
        avg = np.array(new_p_hist).sum(0)/len(new_p_hist)

        # save (if commanded)
        if save:
            self.p = avg
            self.p_hist = new_p_hist

        return avg

    def estimate(self, frame):
        """Estimate the warp parameters that best fit the given frame.

        Note that this implicitly assumes that frames are supplied
        in sequence. This method uses the previously solved for warp 
        parameters of (presumably) the previous frame in the same sequence.
        """
        
        # start with our previous parameter estimate
        p = self.p
        dP = self.p + np.inf

        # precompute anything we can
        frame_gradient = [
            cv2.Sobel(np.float32(frame),cv2.CV_32F,1,0,ksize=3),
            cv2.Sobel(np.float32(frame),cv2.CV_32F,0,1,ksize=3)
        ]

        # begin iteration until gradient descent converges
        count = 0
        st = time.time()
        while np.linalg.norm(dP) > self.epsilon:

            # get representation of affine transform
            W = np.array([[1,0,0],[0,1,0]]) + p
            W = cv2.invertAffineTransform(W)

            # warp image with current parameter estimate
            I = cv2.warpPerspective(cv2.warpAffine(frame,W,self.shape),self.H,self.shape)

            # convert various entities to floating point
            I = np.float32(I)
            grad = np.float32(frame_gradient)

            # scale to match template frame brightness
            #  note: this fails if our previous estimate is bad.
            #        we're probably already in a dead-end, but might as well try.
            if I.mean() != 0:
                scale = self.template.mean()/I.mean()
                I *= scale

            # compute error image
            E = self.template - I

            I_grad = np.array([
                cv2.warpPerspective(cv2.warpAffine(grad[0],W,self.shape),self.H,self.shape),
                cv2.warpPerspective(cv2.warpAffine(grad[1],W,self.shape),self.H,self.shape)
            ])

            # calculate steepest descent matrix
            D1 = I_grad[0].reshape(self.shape[1],self.shape[0],1)*self.J[:,:,0,:]
            D2 = I_grad[1].reshape(self.shape[1],self.shape[0],1)*self.J[:,:,1,:]
            D = D1+D2

            # calculate huber loss matrix
            H_w = 0.5*(E*E)
            H_w[abs(E)>self.sigma] = (self.sigma*abs(E)-0.5*self.sigma)[abs(E)>self.sigma]

            # calculate Hessian and remaining terms needed to solve for dP
            H = np.tensordot(D,H_w.reshape(self.shape[1],self.shape[0],1)*D,axes=((0,1),(0,1)))
            O = (D*(H_w*E).reshape(self.shape[1],self.shape[0],1)).sum((0,1))

            # calculate parameter delta
            try:
                dP = np.linalg.inv(H)@O
            except np.linalg.LinAlgError:
                return False, None

            # update parameter estimates
            p += dP.reshape(3,2).T

            # update our counter and evaluate stop criterion
            count += 1

            # stop after max_count iterations
            if count >= self.max_count:
                return False, self.average(p, save=False)
        
        # we've converged: update current location estimate
        return True, self.average(p)






