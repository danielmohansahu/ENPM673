"""Python implementation of the Lucas-Kanade algorithm for template tracking via Affine Transformation.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2

class LucasKanade:

    def __init__(self, template, bounding_box):
        """Initialize the Lucas-Kanade algorithm.

        Args:
            template:   Image containing the template to track.
            bbox:       Bounding box of the template.
        """

        # remove everything from the template image except the bounding box
        self.template = np.float32(template)
        self.shape = (template.shape[1],template.shape[0])

        # compute homography from template image to ROI
        bb = bounding_box
        tpts = np.array([[bb[0],bb[1]],[bb[0]+bb[2],bb[1]],[bb[0]+bb[2],bb[1]+bb[3]],[bb[0],bb[1]+bb[3]]])
        ipts = np.array([[0,0],[640,0],[640,360],[0,360]])
        self.H = cv2.findHomography(tpts,ipts)[0]

        # get our template
        self.template = cv2.warpPerspective(template, self.H, dsize=self.shape)

        # initialize our parameter estimate (to zero)
        self.p = np.zeros((2,3),dtype=np.float32)

        # initialize certain constant parameters
        self.J = np.zeros((self.shape[1],self.shape[0],2,6))
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                self.J[y,x] = np.array([[x,0,y,0,1,0],[0,x,0,y,0,1]])

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

        # precompute anything we can
        grad = np.gradient(frame)

        # begin iteration until gradient descent converges
        count = 0
        st = time.time()
        while np.linalg.norm(dP) > self.epsilon:
            # warp image with current parameter estimate
            # W = np.linalg.inv(np.vstack((p,np.array([0,0,1]))))
            W = np.array([[1,0,0],[0,1,0]]) + p
            I = cv2.warpPerspective(ndimage.affine_transform(frame, W),self.H,self.shape)

            # compute error image
            E = np.abs(self.template - I)

            # warp current gradient estimate
            I_grad = np.array([
                cv2.warpPerspective(ndimage.affine_transform(grad[0], W), self.H, self.shape),
                cv2.warpPerspective(ndimage.affine_transform(grad[1], W), self.H, self.shape)
            ])

            # calculate steepest descent matrix
            D1 = I_grad[0].reshape(360,640,1)*self.J[:,:,0,:]
            D2 = I_grad[1].reshape(360,640,1)*self.J[:,:,1,:]
            D = D1+D2

            # calculate Hessian and remaining terms needed to solve for dP
            H = np.tensordot(D,D,axes=((0,1),(0,1)))
            O = (D*E.reshape(360,640,1)).sum((0,1))

            # calculate parameter delta
            dP = np.linalg.inv(H)@O

            # update parameter estimates
            p += dP.reshape(3,2).T

            if count == 100:
                pass
                # import pdb;pdb.set_trace()

            count += 1
            if count%25==0:
                print("On iteration {} ({:.3f} seconds so far): dP norm: {:.3f}".format(count, time.time()-st, np.linalg.norm(dP)))

        # we've converged: update internal variables
        self.p = p
        return p






