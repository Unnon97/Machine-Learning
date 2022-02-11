import numpy as np
def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    #####Insert your code here for subtask 6a#####
    N= np.size(X,0)
    D = np.size(X,1)
    means = np.array(means)
    weights = np.array(weights)
    logLikelihood = 0
    pxtheta = np.zeros(N)
    pXn = np.zeros(N)
    
    constant = (2*np.pi)**(-0.5*D)
    
    for i in range(N):
        x = X[i,:].reshape((D,1))
        pxtheta = np.zeros(N)
        for j in range(np.size(means,0)):
            mu = means[j,:].reshape((D,1))
            mat1 = (x-mu).T
            mat2 = np.linalg.inv(covariances[:,:,j])
            mat3 = x-mu
            expval = np.exp( (-0.5)*np.matmul( mat1, np.matmul( mat2,mat3 ) ))
            pxtheta[j] = constant * (np.linalg.det(covariances[:,:,j])**(-0.5)) * (expval) 
            pXn[i] += pxtheta[j]*weights[j] 
        logLikelihood += np.log(pXn[i])
    
    return logLikelihood

