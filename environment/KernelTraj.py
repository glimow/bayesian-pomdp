import math
import numpy as np
from environment.state import State

from tools.KMCF import KMCF

__author__ = 'philippe'


class KernelTraj:
    def __init__(self, ld_params, state, restrict_angle=True):
        """
        Compute a Kernel Trajectory (using KMCF) that goes through a specified set of points. The points are specified
        with a (distance, angle) pair from the previous point, starting from 0.0
        :param ld_params: array of (distance, angle) pairs. Angles are in rad
        :param state: initial state, including initial angle
        :param restrict_angle: adds an additional point at t=0.01 to enforce a starting angle. (default True)
        :return: a KernelTraj object from which a trajectory can be retrieved
        """
        self.starting_state = state
        self.ld_params = ld_params
        if restrict_angle:
            self.restrict_angle = [math.cos(state.w), math.sin(state.w)]
        else:
            self.restrict_angle = None

        # Compute the number of anchor points
        anchor_offset = 1 if restrict_angle else 0
        num_pts = len(ld_params) + 1 + anchor_offset
        self.anchor_pts = np.zeros((num_pts, 2))

        # Create a time vector
        self.times = np.linspace(0, 1, len(ld_params) + 1)[:, np.newaxis]

        # Set the agent's state as first point in the trajectory
        self.anchor_pts[0] = [state.x, state.y]

        # Potentially add another point very close to the starting point, to enforce a starting angle
        if restrict_angle:
            dt = 0.01
            self.times = np.insert(self.times, 1, [dt], axis=0)
            self.anchor_pts[1] = [state.x + dt * math.cos(state.w), state.y + dt * math.sin(state.w)]

        # For now, just use a constant distance of 0.8 for the whole trajectory
        d = 1.0 / (float(len(ld_params)))

        old_a = state.w
        for i in range(len(ld_params)):
            # Convert the angle from [-1, 1]
            if i == 0:
                # Convert the first angle to the interval 0.15*[-pi, pi]
                angle_fac = 0.25
            else:
                # Convert all other angles to the interval 0.50*[-pi, pi]
                angle_fac = 0.40
            a = old_a + angle_fac * np.pi * ld_params[i]

            # Build the anchor point list
            self.anchor_pts[anchor_offset + i + 1] = [self.anchor_pts[anchor_offset + i, 0] + np.cos(a) * d,
                                                      self.anchor_pts[anchor_offset + i, 1] + np.sin(a) * d]
            old_a = a

        # Rescale the anchor points to the interval [0,1]
        pt_min_x = np.min(self.anchor_pts[:, 0])
        pt_max_x = np.max(self.anchor_pts[:, 0])
        pt_min_y = np.min(self.anchor_pts[:, 1])
        pt_max_y = np.max(self.anchor_pts[:, 1])
        rescale_factor_x = max(1.0, pt_max_x - pt_min_x)
        rescale_factor_y = max(1.0, pt_max_y - pt_min_y)
        self.rescale_factor = [rescale_factor_x, rescale_factor_y, pt_min_x, pt_min_y]
        self.anchor_pts = np.array([(self.anchor_pts[:, 0] - pt_min_x) / rescale_factor_x,
                                    (self.anchor_pts[:, 1] - pt_min_y) / rescale_factor_y]).T

    def get_samp_traj(self, steps, time_offset=0):
        """
        Method used to retrieve a sampled trajectory.
        :param steps: The number of steps for the sampling.
        :param time_offset: when using time, world time at which the trajectory starts
        :return: a matrix of positions corresponding to the given times.
        """
        kmcf = KMCF(0.2, 15.0, 0.1, 0.00001)
        # kmcf.set_anchor_pts(self.anchor_pts, self.times, 2 if self.restrict_angle else 1)
        kmcf.set_anchor_pts(self.anchor_pts, self.times, 1, self.restrict_angle)

        samp_times = np.atleast_2d(np.linspace(0.0, 1.0, steps + 1)).T

        raw_coords = kmcf.test_mean(samp_times)

        # Rescale points
        coords = np.array([raw_coords[:, 0] * self.rescale_factor[0] + self.rescale_factor[2],
                           raw_coords[:, 1] * self.rescale_factor[1] + self.rescale_factor[3]]).T

        # Add time
        pos = np.append(coords, samp_times, axis=1)

        # Compute the angle and angular velocity and convert them to states
        dt = 1.0 / (steps + 1)
        states = [None] * len(pos)
        w_last = self.starting_state.w
        states[0] = self.starting_state.clone()
        for i in range(1, len(states)):
            w = math.atan2(pos[i][1] - pos[i - 1][1], pos[i][0] - pos[i - 1][0])
            w_vel = (w - w_last) / (2 * dt)
            states[i] = State(pos[i, 0], pos[i, 1], t=pos[i, 2] + time_offset, w=w, w_vel=w_vel)
            w_last = w

        return states

    def samp_traj_at(self, time, time_offset=0):
        """
        Method used to sample the trajectory at a single location
        :param time: time at which to sample. 0<=t<=1
        :param time_offset: when using time, world time at which the trajectory starts
        :return: a position corresponding to the given time
        """
        kmcf = KMCF(0.2, 15.0)
        # kmcf.set_anchor_pts(self.anchor_pts, self.times, 2 if self.restrict_angle else 1)
        kmcf.set_anchor_pts(self.anchor_pts, self.times, 1, self.restrict_angle)

        dt = 0.01
        raw_pos = kmcf.test_mean(np.array([[0.0], [time - dt], [time], [time + dt]]))

        # Rescale points
        pos = np.array([raw_pos[:, 0] * self.rescale_factor[0] + self.rescale_factor[2],
                        raw_pos[:, 1] * self.rescale_factor[1] + self.rescale_factor[3]]).T

        # Compute the angle and angular velocity
        w1 = math.atan2(pos[2][1] - pos[1][1], pos[2][0] - pos[1][0])
        w2 = math.atan2(pos[3][1] - pos[2][1], pos[3][0] - pos[2][0])
        w_vel = (w2 - w1) / (2 * dt)

        # Convert them to states
        return State(pos[2][0], pos[2][1], w=w1, w_vel=w_vel, t=time + time_offset)


if __name__ == '__main__':
    def jerk(x):
        v = np.gradient(x, axis=0)
        a = np.gradient(v, axis=0)
        j = np.gradient(a, axis=0)
        return np.sum(np.linalg.norm(j, 2, axis=1))


    import matplotlib.pyplot as plt

    plt.subplot(1, 1, 1)
    nparams = 3
    p = State(0, 0, w=0.0, t=0)
    jerks = []
    for _ in range(20):
        params = np.random.uniform(-1, 1, (nparams, 1))
        # p.w = np.random.uniform(-np.pi, np.pi)
        ker_traj = KernelTraj(params, p, True)
        traj = np.array(map(State.to_xy_array, ker_traj.get_samp_traj(100)))

        plt.plot(traj[:, 0], traj[:, 1], 'b-')
        anchors = ker_traj.anchor_pts
        anchors = np.array([anchors[:, 0] * ker_traj.rescale_factor[0] + ker_traj.rescale_factor[2],
                            anchors[:, 1] * ker_traj.rescale_factor[1] + ker_traj.rescale_factor[3]]).T

        jerks.append(jerk(traj))
        # plt.plot(anchors[:, 0], anchors[:, 1], '*r')

    print np.mean(jerks)
    plt.xlim(p.x - 1.1, p.x + 1.1)
    plt.ylim(p.y - 1.1, p.y + 1.1)
    plt.show()
