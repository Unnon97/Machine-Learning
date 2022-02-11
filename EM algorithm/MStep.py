import numpy as np
from getLogLikelihood import getLogLikelihood


def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    #####Insert your code here for subtask 6c#####
    
    gamma = np.array(gamma)
    N = np.size(X,0)
    D = np.size(X,1)
    K = np.size(gamma,1)
    Ntilde = np.zeros((1,K))
    means = np.zeros((K,D))
    weights = np.zeros((1,K))
    covariances = np.zeros((D,D,K))
        
    
    
    Ntilde = np.sum(gamma,axis=0).reshape((1,K))
    weights = (Ntilde/N).reshape((K,))
    means = ((Ntilde**(-1))) * (gamma.T @ X).T 
    means = means.T
    
    for i in range(K):
        mu = means[i,:].reshape((D,1))
        X1 = X.T            
        covariances[:,:,i] = (Ntilde[0,i]**(-1)) * (gamma[:,i]*(X1-mu)) @ (X1-mu).T
    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    
    
    return weights, means, covariances, logLikelihood



