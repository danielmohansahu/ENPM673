"""Python implementation of the Lucas-Kanade algorithm for template tracking via Affine Transformation.
"""

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

        # other variables
        self.epsilon = 0.1

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

        # begin iteration until gradient descent converges
        while np.linalg.norm(dP) > self.epsilon:

            # warp image with current parameter estimate
            W = np.array([[1,0,0],[0,1,0]]) + p.reshape(3,2).T
            I = cv2.warpAffine(frame.T, W, frame.shape)

            # compute error image
            E = self.template.T-I

            # compute gradient and warp
            k_y = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
            k_x = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
            I_grad = np.array([
                cv2.warpAffine(cv2.filter2D(frame,cv2.CV_8U,k_x), W, frame.shape),
                cv2.warpAffine(cv2.filter2D(frame,cv2.CV_8U,k_y), W, frame.shape)
            ])

            # evaluate steepest descent (per-pixel)
            O = np.zeros((1,6)) 
            H = np.zeros((6,6))
            for x in range(I.shape[0]):
                for y in range(I.shape[1]):

                    # Jacobian:
                    J = np.array([[x,0,y,0,1,0],[0,x,0,y,0,1]])

                    # steepest descent:
                    D = I_grad[:,x,y]@J

                    # accumulate Hessian and other term
                    H += np.outer(D,D)
                    O += D.T*E[x,y]

            # calculate parameter delta
            dP = np.linalg.inv(H)@O.T

            # update parameter estimates
            p += dP.T

            count += 1
            print("Iteration {}: dP norm: {:.3f}".format(count,np.linalg.norm(dP)))

        # we've converged: update internal variables
        self.p = p
        return p






