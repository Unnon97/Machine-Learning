import numpy as np


def knn(samples, k):
    # pos = numpy.arange(-5, 5.0, 0.1)
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]

    #####Insert your code here for subtask 5b#####
    samples = np.sort(samples)
    pos = np.arange(-5, 5.0, 0.1)
    distances = np.zeros(np.size(samples))
    vol = np.zeros(np.size(samples))
    N = np.size(pos)
    estDensity = np.zeros((N,2))
    estDensity[:,0] = pos
    
    for i in range(N): 
        distances = np.zeros(np.size(samples))
        for j in range(np.size(samples)):
            distances[j] = abs(pos[i] - samples[j])
        distances = np.sort(distances)
        vol[i] = 2*distances[k+1]
        estDensity[i,1] = k/(N*vol[i])
   
    # Compute the number of the samples created
    return estDensity
