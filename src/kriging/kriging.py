import numpy as np
from scipy.optimize import minimize
from aux import KrigingAux

class Kriging(KrigingAux):
    def __init__(self, kernel="ardmatern32", regularize=True, normalize=True, n_restart=1):
        super().__init__()
        self.kernel = kernel
        self.regularize = regularize
        self.normalize = normalize
        self.n_restart = n_restart
    

    def neg_loglikeli(self, hyparas, x, y):
        '''
        objective function.
        evaluate negative log-likelihood for given hyperparameters.
        '''

        n = x.shape[0]
        d = x.shape[1]
        hyparas = np.asarray(hyparas)

        # transform back to original scale
        lscale = np.exp(hyparas[:d]) # length scale parameter
        if self.regularize:
            reg = np.exp(hyparas[d])**2 # regularaization parameter
        else:
            reg = len(x) * np.finfo(float).eps

        K = self._kernel_mat(x, x, lscale, kernel=self.kernel) # (n,n)
        K += reg * np.eye(len(K)) # add regularization (n,n)

        # cholesky decomposition
        L = np.linalg.cholesky(K) # (n,n)

        # compute log-likelihood: y^T K^-1 y
        a = np.linalg.solve(L, y) # (n,1)
        b = np.linalg.solve(L.T, a) # (n,1)
        elem1 = np.dot(y.T, b) # (1,1)

        # compute process variance
        var = elem1 / n # (1,)

        # compute log-likelihood: log|K|
        diags = np.diag(L) # (n,)
        elem2 = 2 * np.sum(np.log(diags)) # (1,)

        loglikeli = 0.5 * (n* np.log(var) +  elem2)

        return float(loglikeli)


    def train(self, x, y):
        '''
        obtain optimal hyperparameters by minimizing negative log-likelihood.
        '''
        # ---------------------------------------------
        #             set up training data
        # ---------------------------------------------
        x = self._preproc(x) # preprocess data
        y = self._preproc(y)

        if self.normalize:
            x, self.x_mean, self.x_std = self._normalize_data(x) # normalize data
            y, self.y_mean, self.y_std = self._normalize_data(y) # normalize data
            
        else:
            self.x_mean = None
            self.x_std = None
            self.y_mean = None
            self.y_std = None
        
        self.x_train, self.y_train = x, y
        d = x.shape[1]


        # ---------------------------------------------
        #           optimize hyperparameters
        # ---------------------------------------------        
        # bound constraints
        bounds = self.get_bounds(x, d)
        lb = bounds[:,0]
        ub = bounds[:,1]

        # initial hyperparameters
        hyparas0 = self.init_hyparas(bounds, d)

        best_loglikeli = np.inf
        for _ in range(self.n_restart): # repeat optimization n_restart times
            hyparas0 = np.random.uniform(lb, ub) # random initial guess
            sol = minimize(self.neg_loglikeli, hyparas0, args=(self.x_train, self.y_train),
                           method='SLSQP', bounds=bounds)
            if sol.fun < best_loglikeli: # save the best hyperparameters
                best_loglikeli = sol.fun
                best_hyparas = sol.x

        self.opt_hyparas = best_hyparas
        self.lscale = np.exp(self.opt_hyparas[:d]) # scale back
        if self.regularize:
            self.reg = np.exp(self.opt_hyparas[d])**2 # scale back
        else:
            self.reg = len(x) * np.finfo(float).eps


        # ---------------------------------------------
        #       save L, weights for prediction
        # ---------------------------------------------
        # compute kernel matrix btw training points
        K = self._kernel_mat(self.x_train, self.x_train, self.lscale, 
                       kernel=self.kernel) # (n,n)
        K += self.reg * np.eye(len(K)) # add regularization (n,n)

        # cholesky decomposition
        self.L = np.linalg.cholesky(K) # (n,n)

        # compute weights for prediction
        a = np.linalg.solve(self.L, self.y_train) # (n,1)
        self.weight = np.linalg.solve(self.L.T, a) # = (K + reg*I)^-1 * y, (n,1) 

        return self.opt_hyparas
    

    def predict(self, x_test, return_var=False):
        '''
        predict at unseen points.
        compute posterior mean and variance.
        '''
        # preprocess data
        x_test = self._preproc(x_test)
    
        if self.normalize:
            # normalize test data
            x_test = (x_test - self.x_mean) / self.x_std

        # kernel matrix btw training and test points
        k = self._kernel_mat(self.x_train, x_test, self.lscale, kernel=self.kernel) # (n, n_test)

        # compute posterior mean
        y_pred = np.dot(k.T, self.weight)

        if self.normalize:
            y_pred = y_pred * self.y_std + self.y_mean # unnormalize prediction

        if return_var:
            # compute posterior variance
            c = np.linalg.solve(self.L, k) # (n, n_test)
            k_test = self._kernel_mat(x_test, x_test, self.lscale, kernel=self.kernel) # (n_test, n_test)
            k_test = np.diag(k_test) # (n_test,)
            var_pred = k_test - np.sum(c**2, axis=0) # (n_test,)

            if self.normalize:
                var_pred = var_pred * self.y_std**2 # unnormalize prediction variance

            return y_pred, var_pred

        return y_pred