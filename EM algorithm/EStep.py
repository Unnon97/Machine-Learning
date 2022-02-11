import numpy as np
from getLogLikelihood import getLogLikelihood


def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    #####Insert your code here for subtask 6b#####
    weights = np.array(weights)
    means = np.array(means)
    N = np.size(X,0)
    D = np.size(X,1)
    K = np.size(weights)
    gamma = np.zeros((N,K))
    pXn = np.zeros((N,1))
    
    constant = (2*np.pi)**(-0.5*D)
    
    for i in range(N):
        x = X[i,:].reshape((D,1))
        # pxtheta = np.zeros(N)
        for j in range(np.size(means,0)):
            mu = means[j,:].reshape((D,1))
            mat1 = (x-mu).T
            mat2 = np.linalg.inv(covariances[:,:,j])
            mat3 = x-mu
            expval = np.exp( (-0.5)*np.matmul( mat1, np.matmul( mat2,mat3 ) )).squeeze()
            # print(np.linalg.cond(covariances[:, :,j]))
            gamma[i,j] = (weights[j])*constant * (np.linalg.det(covariances[:,:,j])**(-0.5)) * (expval) 
            pXn[i] += gamma[i,j] 
    gamma = gamma/pXn
            
    
    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    
    return [logLikelihood, gamma]
