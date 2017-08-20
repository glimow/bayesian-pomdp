import copy

import math
import numpy as np
import GPy

from environment.state import State

__author__ = 'philippe'


class Belief:
    def __init__(self, mean_fun, cov_fun, restrict_hyper_parameters=None):
        """
        Belief initialization.
        :param mean_fun: GPy mean function.
        :param cov_fun: GPy covariance function.
        :param restrict_hyper_parameters: function on the model, defining restriction on the hyper-parameters.
        """
        self.need_recompute = True
        self.x = np.array([])
        self.y = np.array([])
        self.mf = mean_fun
        self.cf = cov_fun
        self.restrict_hyper_parameters = restrict_hyper_parameters
        self.model = None

    def update(self, _x, _y):
        """
        Updates the belief with one state-observation pair.
        :param _x: state, as a Pose
        :param _y: observation, as a real number
        """
        # print _y, "_y"
        self.need_recompute = True
        if len(self.x) == 0:
            self.x = np.array([State.to_array(_x)])
            self.y = np.array([[_y]])
        else:
            self.x = np.append(self.x, np.array([State.to_array(_x)]), axis=0)
            self.y = np.append(self.y, np.array([[_y]]), axis=0)

    def update_all(self, _x, _y):
        """
        Updates the belief with several state-observation pairs.
        :param _x: iterable of Poses
        :param _y: iterable of observations, as real numbers
        """
        if len(_x) != len(_y):
            raise Exception('Arguments x and y must be the same length.')
        for i in range(len(_x)):
            self.update(_x[i], _y[i])

    def estimate(self, _x):
        """
        Estimates the model value at a given pose, or multiple poses.
        :param _x: Pose, or iterable of Poses, or array of positions, at which to estimate.
        :return: Model estimation, as a real number
        """
        if self.need_recompute:
            self.___recompute___()
        if hasattr(_x, '__iter__'):
            if isinstance(_x[0], Pose):
                mean, var = self.model.predict(np.array(map(State.to_array, _x)))
            else:
                mean, var = self.model.predict(np.array(_x))
        else:
            mean, var = self.model.predict(np.array([State.to_array(_x)]))
        if math.isnan(var[0]):
            var = np.zeros(var.shape)
        return mean.reshape(-1), var.reshape(-1)

    def ___recompute___(self):
        """
        Recomputes the belief. Only useful before estimating new points.
        """
        if self.model is None:
            # print "CF",self.cf
            # print "X",self.x,"Y", self.y
            self.model = GPy.models.GPRegression(self.x, self.y, self.cf)
        else:
            self.model.set_XY(self.x, self.y)
        self.need_recompute = False

    def optimize(self, max_iter=None):
        """
        Optimizes the underlining model.
        :param max_iter: Maximum number of iterations. (default is None)
        """
        if self.need_recompute:
            self.___recompute___()
        if self.restrict_hyper_parameters is not None:
            self.restrict_hyper_parameters(self.model)

        if not max_iter:
            self.model.optimize(messages=False)
        else:
            self.model.optimize(messages=False, max_f_eval=max_iter)
        print self.model
        self.need_recompute = False

    def clone(self):
        """
        Clones the belief.
        :return: a deep copy of this belief.
        """
        new_belief = Belief(self.mf, self.cf, self.restrict_hyper_parameters)
        new_belief.x = np.copy(self.x)
        new_belief.y = np.copy(self.y)
        new_belief.need_recompute = True
        # It seems copying the model before cloning solves problems... why?
        new_belief.model = self.model
        return new_belief
