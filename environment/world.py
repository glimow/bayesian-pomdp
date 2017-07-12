from pylab import *
import math

from environment.pose import Pose

__author__ = 'philippe'

# When using a Space-Time objective function, period of the time phenomenon
dynamic_function_period = 18


class World:
    def __init__(self, pose, obj_fun_type, action_provider, obs_noise_var=0):
        """
        Initializes the World.
        :param pose: the starting Pose.
        :param obj_fun_type: objective function type.Can be 'static' (Space) or 'dynamic' (Space-Time).
        :param action_provider: an ActionProvider.
        :param obs_noise_var: noise variance to add to observations. (default 0)
        """
        self.time = 0.0
        self.obs_noise_var = obs_noise_var
        self.objective_function = None
        self.np_objective_function = None
        self.pose = pose
        pose.t = self.time
        self.action_provider = action_provider
        if obj_fun_type == 'static':
            self.is_dynamic = False
            self.objective_function = World.static_function
            self.np_objective_function = World.np_static_function
        elif obj_fun_type == 'dynamic':
            self.is_dynamic = True
            self.pose.t = self.time
            self.objective_function = World.dynamic_function
            self.np_objective_function = World.np_dynamic_function
        else:
            raise Exception('Objective function type {} not recognized.'.format(obj_fun_type))

    def static_function(self, pose):
        """
        Static objective function.
        :param pose: pose at which to estimate the objective function.
        :return: objective function value.
        """
        x = Pose.to_xy_array(pose)
        return math.exp(-math.pow(x[0] - 4, 2)) * math.exp(-math.pow(x[1] - 1, 2)) + 0.8 * math.exp(
            -math.pow(x[0] - 1, 2)) * math.exp(-math.pow((x[1] - 4) / 2.5, 2)) + 4 * math.exp(
            -math.pow((x[0] - 10) / 5, 2)) * math.exp(-math.pow((x[1] - 10) / 5, 2))

    def dynamic_function(self, pose):
        """
        Dynamic objective function.
        :param pose: pose at which to estimate the objective function.
        :return: objective function value.
        """
        x = Pose.to_xyt_array(pose)
        return math.exp(
            -2.04082 * math.pow(x[0] - 2 - 1.5 * math.sin(2 * math.pi * x[2] / dynamic_function_period), 2)) * \
               math.exp(
                   -2.04082 * math.pow(x[1] - 2 - 1.5 * math.cos(2 * math.pi * x[2] / dynamic_function_period), 2))

    @staticmethod
    def np_static_function(X, Y):
        """
        Static objective function, for plotting purpuses.
        :param X: array of x-coordinates
        :param Y:array of y-coordinates
        :return: objective function values in an array.
        """
        return np.exp(-np.power(X - 4, 2)) * np.exp(-np.power(Y - 1, 2)) + 0.8 * np.exp(
            -np.power(X - 1, 2)) * np.exp(-np.power((Y - 4) / 2.5, 2)) + 4 * np.exp(
            -np.power((X - 10) / 5, 2)) * np.exp(-np.power((Y - 10) / 5, 2))

    @staticmethod
    def np_dynamic_function(X, Y, T):
        """
        Dynamic objective function, for plotting purpuses.
        :param X: array of x-coordinates
        :param Y:array of y-coordinates
        :param T:array of times
        :return: objective function values in an array.
        """
        return np.exp(-2.04082 * np.power(X - 2 - 1.5 * np.sin(2 * np.pi * T / dynamic_function_period), 2)) * \
               np.exp(-2.04082 * np.power(Y - 2 - 1.5 * np.cos(2 * np.pi * T / dynamic_function_period), 2))

    def execute_full_action(self, action, plotter_res=0):
        """
        Executes an action given by the agent.
        :param action: action to execute.
        :param plotter_res: plotting resolution, 0 means the function is not called for plotting purposes. (default 0)
        :return: a list of positions AND the list of corresponding observations.
        """

        # Retrieve the sampled positions
        samp_pos = self.action_provider.sample_trajectory(action, plotter_res)

        # Update the agent's position
        if plotter_res == 0:
            self.pose = self.action_provider.sample_trajectory_at(action, 1)

        # Compute observations
        time = self.time
        samp_obs = [0] * len(samp_pos)
        dt = 1.0 / float(len(samp_pos))
        for i in range(len(samp_pos)):
            time += dt
            if self.is_dynamic:
                # Correct the time, just in case
                samp_pos[i].t = time
                samp_obs[i] = self.objective_function(self, samp_pos[i])
            else:
                samp_obs[i] = self.objective_function(self, samp_pos[i])

        # Don't increment time when calling for plotting purposes
        if plotter_res == 0:
            self.time = time

        # Potentially add noise to observations
        if self.obs_noise_var > 0:
            return samp_pos, samp_obs + np.random.normal(0, self.obs_noise_var, (len(samp_obs),))
        else:
            return samp_pos, samp_obs
