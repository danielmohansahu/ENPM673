"""Python implementation of the Lucas-Kanade algorithm for template tracking via Affine Transformation.
"""
import time
import numpy as np
import cv2

class LucasKanade:

    def __init__(self, template, bb):
        """Initialize the Lucas-Kanade algorithm.

        Args:
            template:   Image containing the template to track.
            bbox:       Bounding box of the template.
        """

        # remove everything from the template image except the bounding box
        self.template = np.zeros(template.shape)
        self.template[bb[0]:bb[0]+bb[2],bb[1]:bb[1]+bb[3]] = template[bb[0]:bb[0]+bb[2],bb[1]:bb[1]+bb[3]]

        # initialize our parameter estimate (to zero)
        self.p = np.zeros((1,6),dtype=np.float32)

        # initialize certain constant parameters
        self.J = np.zeros((template.shape[1],template.shape[0],2,6)) 
        for x in range(template.shape[1]):
            for y in range(template.shape[0]):
                self.J[x,y] = np.array([[x,0,y,0,1,0],[0,x,0,y,0,1]])

        # other variables
        self.epsilon = 0.1
        self.k_y = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        self.k_x = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])

    def estimate(self, frame):
        """Estimate the warp parameters that best fit the given frame.

        Note that this implicitly assumes that frames are supplied
        in sequence. This method uses the previously solved for warp 
        parameters of (presumably) the previous frame in the same sequence.
        """
        
        # start with our previous parameter estimate
        p = self.p
        dP = self.p + np.inf
        count = 0

        # precompute anything we can
        grad = np.array([
            cv2.filter2D(frame, cv2.CV_8U, self.k_x),
            cv2.filter2D(frame, cv2.CV_8U, self.k_y)
        ])

        # begin iteration until gradient descent converges
        while np.linalg.norm(dP) > self.epsilon:
            st = time.time()

            # warp image with current parameter estimate
            W = np.array([[1,0,0],[0,1,0]]) + p.reshape(3,2).T
            I = cv2.warpAffine(frame.T, W, frame.shape)

            # compute error image
            E = self.template.T-I

            # warp currentgradient estimate
            I_grad = np.array([
                cv2.warpAffine(grad[0], W, frame.shape),
                cv2.warpAffine(grad[1], W, frame.shape)
            ])

            # calculate matrices used to solve for dP
            D1 = I_grad[0].reshape(640,360,1)*self.J[:,:,0,:]
            D2 = I_grad[1].reshape(640,360,1)*self.J[:,:,1,:]
            D = D1+D2
            O = (D*E.reshape(640,360,1)).sum((0,1))
            H = sum([sum([np.outer(d,d) for d in d1]) for d1 in D])

            # calculate parameter delta
            dP = np.linalg.inv(H)@O.T

            # update parameter estimates
            p += dP.T

            count += 1
            print("Iteration {} took {}: dP norm: {:.3f}".format(count, time.time()-st, np.linalg.norm(dP)))

        # we've converged: update internal variables
        self.p = p
        return p






