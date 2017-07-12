from environment.KernelTraj import KernelTraj
from environment.Splines2D import Splines2D
from environment.Splines2D import DiscreteSplines2D
from environment.simulator import Simulator

__author__ = 'philippe'


class ActionProvider:
    """
    Interfaces between the behaviors algorithms and the several action types.
    Actions can be discrete splines (disc_spline), continuous splines (cont_spline or cont_spline* to use discrete
    spline initialization), or Kernel trajectories (kernel_traj).
    """

    def __init__(self, boundaries, nparams, type, is_dynamic):
        """
        :param boundaries: Domain boundaries
        :param nparams: Dimension of the expected ld_params
        :param type: Type can be 'cont_sline', 'cont_spline*' 'disc_spline' or 'kernel_traj'
        :type is_dynamic: wether or not the time dimension has to be included in computations.
        """
        self.nparams = nparams
        self.type = type
        self.sampling_step_nb = 5
        if self.type != 'disc_spline' and self.type != 'cont_spline' and \
                        self.type != 'cont_spline*' and self.type != 'kernel_traj':
            raise Exception('Unrecognized action provider type {}'.format(self.type))
        self.simulator = Simulator(boundaries)

    def get_low_dim(self, action):
        """
        Returns a low dimension representation of the action.
        :param action: the action to get a low dimension representation from.
        :return: a list of low dimension parameters.
        """
        return action.ld_params

    def get_boundary_discount(self, p):
        """
        Retrieves the UCB exploration discount for being near the domain boundaries.
        :param p: the Pose or list of Poses at which to compute the discount.
        :return: a discount factor (number between 0 and 1).
        """
        return map(self.simulator.get_boundary_discount, p)

    def sample_trajectory(self, action, sampling_step_nb=0):
        """
        Samples the given trajectory at multiple times.
        :param action: action to sample.
        :param sampling_step_nb: Number of samples in trajectory, 0 means using default class value. (default 0)
        :return: a list of Poses.
        """
        if sampling_step_nb == 0:
            sampling_step_nb = self.sampling_step_nb
        if self.type == 'disc_spline' or self.type == 'cont_spline' or self.type == 'cont_spline*':
            return self.simulator.sample_array_trajectory(action, sampling_step_nb)
        elif self.type == 'kernel_traj':
            return map(self.simulator.restrict_to_bounds, action.get_samp_traj(sampling_step_nb))

    def sample_trajectory_at(self, action, time):
        """
        Samples the given trajectory at a specific time.
        :param action: action to sample.
        :param time: time at which to sample.
        :return: a Pose.
        """
        if self.type == 'disc_spline' or self.type == 'cont_spline' or self.type == 'cont_spline*':
            return self.simulator.sample_trajectory_at(action, time)
        elif self.type == 'kernel_traj':
            return self.simulator.restrict_to_bounds(action.samp_traj_at(time))

    def buid_action_from_params(self, ld_params, pose):
        """
        Constructs an action from low dimension parameters.
        :param ld_params: a list of low dimensions parameters.
        :param pose: a starting Pose.
        :return: an action.
        """
        if self.type == 'disc_spline':
            return DiscreteSplines2D.get_splines(pose)
        elif self.type == 'cont_spline' or self.type == 'cont_spline*':
            return Splines2D.gen_spline_from_lowdim(ld_params, pose)
        elif self.type == 'kernel_traj':
            return KernelTraj(ld_params, pose)
