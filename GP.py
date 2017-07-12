import numpy as np

__author__ = 'philippe'


class GP:
    # TODO: Clean and add hyperparameter optimization
    def __init__(self, cov_fun=None, var_noise=0.01):
        self.x = None
        self.y = np.array([])
        self.K = None
        self.L = None
        self.var_noise = var_noise  # Noise variance
        self.alpha = []
        self.recompute_alpha = True

        if cov_fun is None:
            def __cov_fun(_x, _y):
                return np.exp(-2.0 * np.sum(np.square(_x - _y), axis=1))

            self.__cov_fun = __cov_fun
        else:
            self.__cov_fun = cov_fun

    def update(self, _x, _y):
        # Regular update (when less that 2 points available)
        if len(self.y) < 2:
            self.y = np.append(self.y, _y)
            if self.x is None:
                self.x = np.array([_x])
            else:
                self.x = np.append(self.x, np.array([_x]), 0)

            n = len(self.x)
            self.K = np.reshape(self.__cov_fun(np.repeat([self.x], n, axis=0).reshape(
                (n * n, -1)), np.repeat(self.x, n, axis=0)), (n, n))
            self.L = np.linalg.cholesky(self.K + self.var_noise * np.eye(len(self.x), len(self.x)))
        else:
            # Optimized update
            self.y = np.append(self.y, _y)
            self.x = np.append(self.x, np.array([_x]), 0)

            k_vec = self.__cov_fun(np.repeat([self.x[-1]], len(self.x), axis=0), self.x)

            self.K = np.append(np.append(self.K, [k_vec[:-1]], 0), np.transpose([k_vec]), 1)

            c = np.linalg.solve(self.L, self.K[:-1, -1]).reshape((1, len(self.x) - 1))
            col_vec = np.zeros((len(self.x), 1))
            col_vec[-1] = np.sqrt(self.var_noise + self.K[-1, -1] - np.dot(c, np.transpose(c)))
            self.L = np.append(np.append(self.L, c, 0), col_vec, 1)

        self.recompute_alpha = True

    def estimate(self, _x):
        if self.recompute_alpha:
            self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))
            self.recompute_alpha = False

        # Compute the mean
        if (isinstance(_x, np.ndarray) and _x.ndim > 1) or (isinstance(_x, list) and isinstance(_x[0], list)):
            # TODO: This is not optimized
            ret = np.array([self.__estimate_one(np.atleast_2d(xi)) for xi in _x])
            return ret[:, 0], ret[:, 1]
        else:
            return self.__estimate_one(np.array([_x]))

    # _x must be a np array of dim 2. Each data point is a row
    def __estimate_one(self, _x):
        # Compute the mean
        kp_vec = self.__cov_fun(np.repeat(_x, len(self.x), axis=0), self.x)

        mean_y = np.dot(kp_vec, self.alpha)

        # Compute the variance
        v = np.linalg.solve(self.L, kp_vec)
        var_y = self.__cov_fun(_x, _x) - np.dot(v, v)
        return mean_y, var_y

    def clone(self):
        cln = GP()
        cln.x = np.copy(self.x)
        cln.y = np.copy(self.y)
        cln.K = np.copy(self.K)
        cln.L = np.copy(self.L)
        cln.var_noise = self.var_noise
        cln.alpha = np.copy(self.alpha)
        cln.recompute_alpha = self.recompute_alpha
        return cln
