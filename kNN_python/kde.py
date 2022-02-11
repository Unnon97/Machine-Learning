import numpy as np

def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    #####Insert your code here for subtask 5a#####
    pos = np.arange(-5, 5.0, 0.1)
    N = np.size(pos)
    estDensity = np.zeros((N,2))
    estDensity[:,0] = pos
    
    for i in range(N):
        for j in range(np.size(samples)):
            norm = pos[i] - samples[j]
            exponentval = (norm**2) /(2*(h**2))
            constants = ((N * h * (2*np.pi)**0.5))
            estDensity[i,1] += (np.exp(0-exponentval)) / constants

    # Compute the number of samples created
    return estDensity
