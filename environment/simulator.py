import numpy as np
from environment.pose import Pose

__author__ = 'philippe'


class Simulator:
    """
    Simulates boundaries conditions so that the trajectories stay within the permitted area.
    """

    def __init__(self, boundaries):
        """
        Initializes simulator.
        :param boundaries: Domain boundaries.
        """
        self.boundaries = boundaries
        self.boundary_discount_dist = 0.5
        self.boundary_discount_min = 0.5
        self.boundary_discount_max = 1.0

    # This computes a factor to apply to the GP uncertainty when computing the acquisition function UCB,
    # so that being close to the boundaries of the domain is penalized. This is done so that the agent does not expect
    # to reduce the uncertainty outside of the domain.
    def get_boundary_discount(self, p):
        """
        This function computes a factor to apply to the GP uncertainty when computing the acquisition function UCB,
        so that being close to the boundaries of the domain is penalized. This is done so that the agent does not expect
        to reduce the uncertainty outside of the domain.
        :param p: the Pose for which to compute the boundary factor.
        :return: the boundary factor (real number).
        """
        x_d = min(p.x - self.boundaries[0][0], self.boundaries[0][1] - p.x)
        y_d = min(p.y - self.boundaries[1][0], self.boundaries[1][1] - p.y)
        d = max(0, min(x_d, y_d, self.boundary_discount_dist))
        return (d / self.boundary_discount_dist) * (
            self.boundary_discount_max - self.boundary_discount_min) + self.boundary_discount_min

    def sample_array_trajectory(self, act, steps):
        """
        Samples a trajectory and restricts it to the bounds of the domain.
        :param act: action to sample and restrain.
        :param steps: number of samples.
        :return: an array of Poses.
        """
        dt = np.linspace(0, 1, steps + 1)
        return np.array(map(lambda t: self.sample_trajectory_at(act, t), dt[1:]))

    def sample_trajectory_at(self, act, dt):
        """
        Samples a trajectory at a specific time and restricts it to the bounds of the domain.
        :param act: action to sample and restrain.
        :param dt: time at which to sample the trajectory (real number).
        :return: a Pose.
        """
        new_pose = act.pos_at(dt)
        new_pose.t = dt

        # Restrict pos to bounds
        self.restrict_to_bounds(new_pose)

        return new_pose

    def restrict_to_bounds(self, pose):
        """
        Restricts a Pose to the bounds of the domain.
        :param pose: the Pose to restrict (modified).
        :return: the restricted pose.
        """
        pose.x = max(min(self.boundaries[0][1], pose.x), self.boundaries[0][0])
        pose.y = max(min(self.boundaries[1][1], pose.y), self.boundaries[1][0])
        return pose


if __name__ == '__main__':
    from environment.Splines2D import DiscreteSplines2D

    sim = Simulator([[0, 5], [0, 5]])
    act = DiscreteSplines2D.get_splines(Pose(1, 1, (1, 1), (0, 0)))[0]
    print map(Pose.to_xy_array, sim.sample_array_trajectory(act, 10))
