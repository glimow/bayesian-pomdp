import nlopt
import time

import GPy
from pylab import *

from GP import GP
from tools import putils
from tools.nostdout import nostdout

__author__ = 'philippe'


def dist(a, b):
    return sum([math.pow(x1 - x2, 2) for x1, x2 in zip(a, b)])


class BO:
    def __init__(self, ndim, k, search_int, opt_maxeval=10, steps_btw_opt=20, custom_gp=False, rbf_lengthscale=5.0,
                 rbf_variance=20.0):
        """
        Initialization of BO.
        :param ndim: Number of dimensions.
        :param k: exploration-exploitation UCB parameter.
        :param search_int: search inverval for all dimensions. Should be a 2-element list.
        :param opt_maxeval: Maximum number of evaluations used when optimizing the acquisition function. (default 10)
        :param steps_btw_opt: Number of steps between every BO optimization. (default 20)
        :param custom_gp: Whether or not to use a custom GP. Using GPy othewise. (default False)
        """
        self.__acq_fun = putils.UCB(k)
        self.search_min = search_int[0]
        self.search_max = search_int[1]
        self.d = ndim
        self.last_x_diff = sys.float_info.max
        self.last_x = None
        self.__recompute_x_diff = True
        self.x = []
        self.y = []
        self.custom_gp = custom_gp
        self.steps_btw_opt = steps_btw_opt

        # What GP model to use
        if not self.custom_gp:
            self.cf = GPy.kern.RBF(self.d, variance=rbf_variance, lengthscale=rbf_lengthscale)
            self.gp = None
        else:
            self.gp = GP()

        # Prepare acquisition function
        if custom_gp:
            def __acq_fun_maximize(_x, grad):
                for xi in _x:
                    if xi < self.search_min or xi > self.search_max:
                        return 0.0
                vals = self.gp.estimate(_x)
                return float(self.__acq_fun(vals[0], np.sqrt(vals[1])))
        else:
            def __acq_fun_maximize(_x, grad):
                for xi in _x:
                    if xi < self.search_min or xi > self.search_max:
                        return 0.0

                vals = self.gp.predict(np.array([_x]))
                return float(self.__acq_fun(vals[0], np.sqrt(vals[1])))

        # Prepare acquisition function optimization algorithm
        self.opt = nlopt.opt(nlopt.LN_COBYLA, self.d)
        self.opt.set_lower_bounds(self.search_min)
        self.opt.set_upper_bounds(self.search_max)
        self.opt.set_maxeval(opt_maxeval)
        self.opt.set_max_objective(__acq_fun_maximize)

    def next_sample(self):
        """
        Retrieve the next value at which to sample the objective function.
        :return: next value at which to sample the objective function.
        """
        if len(self.x) <= 1:  # 2 first points are random
            new_x = np.array([np.random.uniform(self.search_min, self.search_max) for _ in range(self.d)])
        else:
            # print "X",self.x[-1]
            new_x = self.opt.optimize(np.array(self.x[-1]))
        self.__compute_x_diff__(new_x)
        return new_x

    def update(self, _x, _y):
        """
        Update the inner model with a new (x,y) observation.
        :param _x: observation location
        :param _y: observation value
        """
        # Add x and y to the dataset
        # print "_x", _x
        # _x = _x[0]
        # print "_x", _x, "self.x", self.x
        if len(self.x) == 0:
            self.x.append(_x)
            self.y.append(_y)
            self.x = np.array(self.x)
            self.y = np.reshape(np.array(self.y), (1, 1))
        else:
            if isinstance(_x, list) or _x.ndim == 1:
                self.x = np.append(self.x, [_x], axis=0)
            else:
                self.x = np.append(self.x, _x, axis=0)
            if isinstance(_y, (int, long, float)):
                self.y = np.append(self.y, [[_y]], axis=0)
            elif isinstance(_y, list) or _y.ndim == 1:
                self.y = np.append(self.y, [_y], axis=0)
            else:
                self.y = np.append(self.y, _y, axis=0)

        # Now, update the gp model
        if self.custom_gp:
            self.gp.update(_x, _y)
        else:
            self.gp = GPy.models.GPRegression(self.x, self.y, self.cf)
            # self.gp['.*rbf.variance'].constrain_bounded(0.0, 50.0, warning=False)
            if len(self.y) == 3 or (len(self.y) + 1) % self.steps_btw_opt == 0:
                with nostdout():
                    self.gp['.*rbf.lengthscale'].constrain_bounded(0.1, 1.0, warning=False)
                    self.gp['.*Gaussian_noise.variance'].constrain_bounded(0, 0.02, warning=False)
                    self.gp.optimize(messages=False, max_f_eval=20)
                #print self.gp

        self.__recompute_x_diff = True

    def __compute_x_diff__(self, new_x):
        """
        Used to compute the distance between the last sampling location and the new sampling location.
        :param new_x: new sampling location.
        """
        if self.__recompute_x_diff:
            self.__recompute_x_diff = False
            if self.last_x is not None:
                self.last_x_diff = dist(self.last_x, new_x)
            self.last_x = new_x


if __name__ == '__main__':
    def np_objective_function(X, Y):
        return np.exp(-np.power(X - 4, 2)) * np.exp(-np.power(Y - 1, 2)) + 0.8 * np.exp(
            -np.power(X - 1, 2)) * np.exp(-np.power((Y - 4) / 2.5, 2)) + 4 * np.exp(
            -np.power((X - 10) / 5, 2)) * np.exp(-np.power((Y - 10) / 5, 2))


    def np_objective_function_1d(X):
        return np.exp(-np.power((X - 1.5) / 2.0, 2) / 0.2) + 3 * np.exp(-np.power((X - 3.5) / 1.0, 2) / 0.2)


    def plot_1d():
        x_min = 0
        x_max = 5
        obj_f_res = 100
        x = np.linspace(x_min, x_max, obj_f_res)[:, None]
        if bo.custom_gp:
            y = np.array([bo.gp.estimate(xi) for xi in x])[:, 0]
        else:
            y = bo.gp.predict(x)[0]

        plt.subplot(1, 1, 1)
        plot(x, y, '-b')
        xlim(x_min, x_max)
        plt.show()


    def plot_2d():
        x_min = 0
        x_max = 5
        y_min = 0
        y_max = 5
        obj_f_res = 20
        X, Y = np.meshgrid(np.linspace(x_min, x_max, obj_f_res), np.linspace(y_min, y_max, obj_f_res))
        Z = np_objective_function(X, Y)
        Z_mean = np.zeros(X.shape)
        Z_std_dev = np.zeros(X.shape)
        if bo.custom_gp:
            for i in range(len(X)):
                for j in range(len(X[0])):
                    Z_mean[i][j], Z_std_dev[i][j] = bo.gp.estimate([[X[i][j], Y[i][j]]])
        else:
            shp = X.shape
            x_vec = np.concatenate((np.reshape(X, (shp[0] * shp[1], 1)), np.reshape(Y, (shp[0] * shp[1], 1))),
                                   axis=1)
            vals = bo.gp.predict(x_vec)
            Z_mean = np.reshape(vals[0], (shp[0], shp[1]))
            Z_std_dev = np.reshape(vals[1], (shp[0], shp[1]))

        samps = np.array(x_history)
        plt.subplot(2, 2, plot_id + 1)
        contourf(X, Y, Z_mean, 16, alpha=.75, cmap='jet')
        contour(X, Y, Z_mean, 16, colors='black', linewidth=.2)
        plot(samps[:, 0], samps[:, 1], '*r')
        xlim(x_min, x_max)
        ylim(y_min, y_max)

        plt.subplot(2, 2, plot_id + 2)
        contourf(X, Y, Z_std_dev, 8, alpha=.75, cmap='jet')
        contour(X, Y, Z_std_dev, 8, colors='black', linewidth=.2)
        xlim(x_min, x_max)
        ylim(y_min, y_max)

        plt.subplot(2, 2, plot_id + 3)
        contourf(X, Y, Z, 8, alpha=.75, cmap='jet')
        contour(X, Y, Z, 8, colors='black', linewidth=.2)
        xlim(x_min, x_max)
        ylim(y_min, y_max)

        # plot_id += 4
        if enable_plot:
            show()


    plot_id = 0
    for custom_gp_val in [False]:
        dim = 2
        start_time = time.time()
        bo = BO(dim, 100, [0.0, 5.0], opt_maxeval=10, custom_gp=custom_gp_val)
        x_history = []
        print 'Step',
        for i in range(40):
            if (i % 10) == 0:
                print i,
            x = bo.next_sample()
            x_history.append(x)
            if dim == 1:
                y = np_objective_function_1d(x)
            else:
                y = np_objective_function(x[0], x[1])
            bo.update(x, y + np.random.normal(0, 0.1))
        print 'Time for', ('library GP', 'custom GP')[custom_gp_val], 'is:', (time.time() - start_time), 'seconds.'

        enable_plot = True
        if enable_plot:
            if dim == 2:
                plot_2d()
            elif dim == 1:
                plot_1d()
