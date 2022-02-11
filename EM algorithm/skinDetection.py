import numpy as np
from estGaussMixEM import estGaussMixEM
from getLogLikelihood import getLogLikelihood



def skinDetection(ndata, sdata, K, n_iter, epsilon, theta, img):
    # Skin Color detector
    #
    # INPUT:
    # ndata         : data for non-skin color
    # sdata         : data for skin-color
    # K             : number of modes
    # n_iter        : number of iterations
    # epsilon       : regularization parameter
    # theta         : threshold
    # img           : input image
    #
    # OUTPUT:
    # result        : Result of the detector for every image pixel

    #####Insert your code here for subtask 1g#####
    

    weights_n, means_n, covariances_n = estGaussMixEM(ndata, K, n_iter, epsilon)
    weights_s, means_s, covariances_s = estGaussMixEM(sdata, K, n_iter, epsilon)
    h,w,d = img.shape
    K = np.size(weights_n,0)
    constant = (2*np.pi)**(-0.5*d)
    Pxtheta_s = np.zeros((h,w))
    Pxtheta_n = np.zeros((h,w))
    
    for i in range(h):
        for j in range(w):
            X = img[i,j,:].reshape((d,1))
            for k in range(K):
                #skindata
                mu_s = means_s[k,:].reshape((d,1))
                mat1_s = (X-mu_s).T
                mat2_s = np.linalg.inv(covariances_s[:,:,k])
                mat3_s = X-mu_s
                expval_s = np.exp( (-0.5)*np.matmul( mat1_s, np.matmul( mat2_s,mat3_s ) )).squeeze()
                Pxtheta_s[i,j] += (weights_s[k])*constant * (np.linalg.det(covariances_s[:,:,k])**(-0.5)) * (expval_s) 
                
                #nonskindata
                mu_n = means_n[k,:].reshape((d,1))
                mat1_n = (X-mu_n).T
                mat2_n = np.linalg.inv(covariances_n[:,:,k])
                mat3_n = X-mu_n
                expval_n = np.exp( (-0.5)*np.matmul( mat1_n, np.matmul( mat2_n,mat3_n ) )).squeeze()
                Pxtheta_n[i,j] += (weights_n[k])*constant * (np.linalg.det(covariances_n[:,:,k])**(-0.5)) * (expval_n) 
                
    result = ((Pxtheta_s/Pxtheta_n) > theta).astype(int)
    
    
    return result
