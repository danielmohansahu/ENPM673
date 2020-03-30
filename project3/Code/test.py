"""Test our ExpectationMaximization algorithm.
"""
import random
import numpy as np
from matplotlib import pyplot as plt
from custom.maximize import ExpectationMaximization as EM

NUM_GAUSSIANS=3
NUM_POINTS=100

if __name__=="__main__":
    K = NUM_GAUSSIANS
    N = NUM_POINTS
    
    # generate a random distribution of points from K gaussians
    mus = [1, -5, 5]
    sigmas = [random.randint(1,5) for k in range(K)]
    X = [[random.gauss(mus[k],sigmas[k]) for i in range(N)] for k in range(K)]
    X = np.array(X).flatten()

    # attempt to solve for those values via Expectation Maximization
    em = EM(K,X)
    print(sigmas)
    mu_guess,cov_guess = em.solve()

    print("Input \tmu: {} \n\tsig: {} \nOutput \tmu: {} \n\tsig: {}".format(
        mus,
        sigmas,
        mu_guess.tolist(),
        np.sqrt(cov_guess).tolist()
    ))

