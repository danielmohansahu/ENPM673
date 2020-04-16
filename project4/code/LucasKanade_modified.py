"""Python implementation of the Lucas-Kanade algorithm for template tracking via Affine Transformation.
"""
import time
import numpy as np
import cv2

class LucasKanade:   # lk = LucasKanade(images[0],TEMPLATE_BBOX)

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
        self.p = np.array([[1,0,0],[0,1,0]],dtype=np.float32)

        # initialize certain constant parameters :dw/dp
        self.J = np.zeros((template.shape[1],template.shape[0],2,6)) 
        
        for x in range(template.shape[1]):
            for y in range(template.shape[0]):
                # dW/dP
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
        print("P initial",p)
        dP = self.p + np.inf

        # precompute anything we can
        grad = np.array([
            cv2.filter2D(frame, cv2.CV_8U, self.k_x),
            cv2.filter2D(frame, cv2.CV_8U, self.k_y)
        ])

        # begin iteration until gradient descent converges
        count = 0
        st = time.time()
        while np.linalg.norm(dP) > self.epsilon:
            # warp image with current parameter estimate
            #W = p
            p = p.reshape(6,1)
            W = np.array([[1 + float(p[0]), float( p[2]), float(p[4])],[float(p[1]), 1 + float(p[3]), float(p[5])]])
            print("W", W)
            #print("value of p[0]", float(p[0]))
            I = cv2.warpAffine(frame.T, W, frame.shape)
           
            # compute error image
            E = frame.T-I
            #print("E", np.shape(E))
            #print("frame transpose",np.shape(frame.T))

            # warp currentgradient estimate
            I_grad = np.array([
                cv2.warpAffine(grad[0], W, frame.shape),
                cv2.warpAffine(grad[1], W, frame.shape)
            ])
            #print("size I_grad", np.shape(I_grad))

            # calculate steepest descent matrix
            D1 = I_grad[0].reshape(640,360,1)*self.J[:,:,0,:]
            #print("D1", D1)
            D2 = I_grad[1].reshape(640,360,1)*self.J[:,:,1,:]
            D = D1+D2
            #print("D", D)

            # calculate Hessian and remaining terms needed to solve for dP
            H = np.tensordot(D,D,axes=((0,1),(0,1)))
            O = (D*E.reshape(640,360,1)).sum((0,1))

            # calculate parameter delta
            dP = np.linalg.inv(H)@O.T
            print("dP",dP)
            
            # update parameter estimates
            p = p.reshape(2,3)
            p += dP.reshape(3,2).T
            print("p", p)
            #print("reshaped p", p)
            count += 1
            if count%500==0:
                print("On iteration {} ({:.3f} so far): dP norm: {:.3f}".format(count, time.time()-st, np.linalg.norm(dP)))
                '''print("D1", np.shape(D1))
                print("I_grad", np.shape(I_grad))
                print("H", np.shape(H))
                print("O", np.shape(O))
                print("lk p",p)
               '''
                
        # we've converged: update internal variables
        self.p = p
        return p


