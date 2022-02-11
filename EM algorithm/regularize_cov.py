import numpy as np


def regularize_cov(covariance, epsilon):
    # regularize a covariance matrix, by enforcing a minimum
    # value on its singular values. Explanation see exercise sheet.
    #
    # INPUT:
    #  covariance: matrix
    #  epsilon:    minimum value for singular values
    #
    # OUTPUT:
    # regularized_cov: reconstructed matrix

    #####Insert your code here for subtask 6d#####
    I = np.identity(np.size(covariance,0))
    regularized_cov = covariance - (epsilon * I)
    return regularized_cov
