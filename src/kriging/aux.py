import numpy as np
from kernels import KERNELS
from scipy.spatial.distance import pdist

class KrigingAux:
    '''
    Defines auxiliary functions for Kriging.
    '''

    def _preproc(self, x):
        # ensure data in 2D array
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        assert x.ndim == 2
        return x


    def _kernel_mat(self, Z1, Z2, lscale, kernel):
        # compute kernel matrix
        return KERNELS[kernel](Z1, Z2, lscale)


    def _normalize_data(self, x):

        x = np.asarray(x)
        if x.ndim == 1: # ensure data in 2D array
            x = x.reshape(-1, 1)
        # get mean and std of data
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
        
        # normalize data
        x_norm = (x - x_mean) / x_std
        
        return x_norm, x_mean, x_std
    

    def init_hyparas(self, bounds, d):
        '''
        inputs
            - bounds: (lower bound, upper bound) in log space for hyperparameters
            - d: input dimension
        '''

        bounds = np.asarray(bounds)
        hyparas0 = np.mean(bounds, axis=1) # (d+1,)
        
        return hyparas0        

    def get_bounds(self, x, d):
        '''
        Set bound constraints for hyperparameters.
        Adapted from pyToolBox's hyperparameter initialization 
        (downloadable from https://link.springer.com/article/10.1007/s00158-022-03274-1#Sec23).
        
        inputs
            - x: training data (n, d)
            - d: input dimension
        outputs
            - bounds: log-scaled (lower bound, upper bound)
        '''
        x = np.asarray(x)

        r = pdist(x, 'euclidean')
        r = r[r > 0] # avoid zero distances
        rmin = np.min(r)
        rmax = np.max(r)

        # set lengthscale bounds
        lb_lscale = np.log(rmin / 10)
        ub_lscale = np.log(10 * rmax)
        length_bounds = np.vstack([ [lb_lscale, ub_lscale] for _ in range(d) ])

        # set noise bounds
        reg_lb = np.log(1e-5)
        reg_ub = np.log(1e-1)
        reg_bounds = np.array([[reg_lb, reg_ub]])

        # combine bounds
        bounds = np.vstack([length_bounds, reg_bounds])

        return bounds
