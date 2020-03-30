"""Implementation of the Expectation Maximization Algorithm for a Gaussian distribution."""

import random
import numpy as np
from scipy.stats import multivariate_normal
from numpy.linalg import det, inv

# solver parameters; max iterations and minimum per-loop delta 
MAX_ITERATIONS=10000
MIN_LIKELIHOOD_CHANGE=0.0001

class ExpectationMaximization:
    def __init__(self, K, X):
        # set inputs as class members
        self.X = X
        self.K = K
        if X.ndim == 1:
            self.M,self.N = 1,len(X)
        else:
            self.M,self.N = X.shape

        # initialize cluster variables (mean, covariance, weights)
        self.mu = np.array([random.random() for k in range(K)])
        self.wgt = np.ones(K)/K
        self.cov = np.array(K*[np.eye(self.M)])
        self.resp = np.zeros((K,self.N))

        # iteration parameters (stop criteria)
        self._max_iterations = MAX_ITERATIONS
        self._epsilon = MIN_LIKELIHOOD_CHANGE

    def solve(self):
        """Iteratively maximize the likelihood function (e.g. solve
        for best mean/variance).
        """
        # initialize loop variables
        count = 0
        like = np.inf
        prev_like = self._likelihood()

        while abs(like-prev_like) > self._epsilon:
            # evaluate stop condition
            if count > self._max_iterations:
                raise RuntimeError("Failed to converge.")

            # call expectation stage
            self._expectation()
            
            # call maximization stage
            self._maximization()
            
            # recalculate likelihood
            prev_like = like
            like = self._likelihood()
            count += 1

        # return our calculated gaussian distribution parameters
        return self.mu.squeeze(), self.cov.squeeze()

    def _expectation(self):
        # Employ Baye's rule to re-calculate the responsibilites 
        #  based on current parameters. 
        # Responsibilities are posterior probabilities.
        
        norms = [multivariate_normal(self.mu[k],self.cov[k],allow_singular=True) for k in range(self.K)]
        for i,x in enumerate(self.X):
            total_prob = sum([self.wgt[k]*norms[k].pdf(x) for k in range(self.K)])
            for k in range(self.K):
                self.resp[k][i] = self.wgt[k]*norms[k].pdf(x)/total_prob

    def _maximization(self):
        # re-calculate the current parameters based on responsibilities
        for k in range(self.K):
            pts = sum(self.resp[k])
            self.mu[k] = sum(x*self.resp[k][i] for i,x in enumerate(self.X))/pts
            self.cov[k] = sum(self.resp[k][i]*np.dot(x-self.mu[k],x-self.mu[k]) for i,x in enumerate(self.X))/pts
            self.wgt[k] = pts/self.N

    def _likelihood(self):
        # calculate the current log likelihood of our parameters
        norms = [multivariate_normal(self.mu[k],self.cov[k],allow_singular=True) for k in range(self.K)]
        res = 0
        for x in self.X:
            res += sum([self.wgt[k]*norms[k].pdf(x) for k in range(self.K)])
        res = np.log(res)
        return res








