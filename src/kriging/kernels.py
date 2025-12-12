import numpy as np
from scipy.spatial.distance import cdist

'''
Stores kernel functions
Reference: 
    kernels: https://www.mathworks.com/help/stats/kernel-covariance-function-options.html
'''

def expkernel(Z1,Z2,lscale=0.1):
    '''
    exponential kernel function
    inputs:
        Z1: input 2D array (N1,d_in)
        Z2: input 2D array (N2,d_in)
    lscale: length scale parameter (scalar)

    outputs:
        K: kernel matrix (N1,N2)
    '''
    dist = cdist(Z1,Z2,'sqeuclidean') # dist[i][j] = distance btw Z1[i] and Z2[j]
    K = np.exp(-dist/lscale)

    return K

def ardmatern32(Z1,Z2,lscale=0.1):
    '''
    ARD Matern 3/2 kernel function

    inputs:
        Z1: input 2D array (N1,d_in)
        Z2: input 2D array (N2,d_in)
    lscale: length scale parameter (scalar) if d_in=1 with single output
        or (d_in,) if ard kernel with d_in inputs and sigle output
        or (d_in,r) if ard kernel with d_in inputs and multi-output

    outputs:
        K: kernel matrix (N1,N2)
    '''
    # ensure data in 2D array
    if Z1.ndim == 1:
        Z1 = Z1.reshape(-1, 1)
    if Z2.ndim == 1:
        Z2 = Z2.reshape(-1, 1)

    d = Z1.shape[1] # number of inputs (d_in)
    lscale = np.asarray(lscale)
    if lscale.ndim == 0: # if lscale is scalar, use the same lscale across the inputs (d_in,)
        lscale = np.repeat(lscale.item(),d)
    elif lscale.ndim == 1 and lscale.size == 1:
        lscale = np.repeat(lscale.item(),d)

    dist = cdist(Z1,Z2,'seuclidean',V=lscale**2)
    K = (1 + np.sqrt(3) * dist)*np.exp(-np.sqrt(3) * dist)

    return K # (N1,N2)

KERNELS = {
    'exp': expkernel,
    'ardmatern32': ardmatern32
}