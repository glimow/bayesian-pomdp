import numpy as np
from state import State

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
    def get_boundary_discount(self, state):
        """
        This function computes a factor to apply to the GP uncertainty when computing the acquisition function UCB,
        so that being close to the boundaries of the domain is penalized. This is done so that the agent does not expect
        to reduce the uncertainty outside of the domain.
        :param s: the State for which to compute the boundary factor.
        :return: the boundary factor (real number).
        """
        distances = []
        for coord in range(len(state.x)):
            distances.append(min(state.x[coord] - self.boundaries[coord][0], self.boundaries[coord][1] - state.x[coord]))

        #y_d = min(p.y - self.boundaries[1][0], self.boundaries[1][1] - p.y)
        d = max(0, min(min(distances), self.boundary_discount_dist))
        return (d / self.boundary_discount_dist) * (
            self.boundary_discount_max - self.boundary_discount_min) + self.boundary_discount_min

    def sample_array_trajectory(self, act, steps):
        """
        Samples a trajectory and restricts it to the bounds of the domain.
        :param act: action to sample and restrain.
        :param steps: number of samples.
        :return: an array of states.
        """
        dt = np.linspace(0, 1, steps + 1)
        return np.array(map(lambda t: self.sample_trajectory_at(act, t), dt[1:]))

    def sample_trajectory_at(self, act, dt):
        """
        Samples a trajectory at a specific time and restricts it to the bounds of the domain.
        :param act: action to sample and restrain.
        :param dt: time at which to sample the trajectory (real number).
        :return: a state.
        """
        new_state = act.pos_at(dt)
        new_state.t = dt

        # Restrict pos to bounds
        self.restrict_to_bounds(new_state)

        return new_state

    def restrict_to_bounds(self, state):
        """
        Restricts a state to the bounds of the domain.
        :param state: the state to restrict (modified).
        :return: the restricted state.
        """
        for index, x in enumerate(state.x):
            # print index, state.x, self.boundaries
            x = max(min(self.boundaries[index][1], x), self.boundaries[index][0])
        #state.y = max(min(self.boundaries[1][1], state.y), self.boundaries[1][0])
        return state


if __name__ == '__main__':
    from Action import Action

    sim = Simulator([[0, 5], [0, 5]])
    act = Action([(1,2),(3,4)],State([0,1]));
    print sim.sample_array_trajectory(act, 10)
    # print map(State.to_xy_array, sim.sample_array_trajectory(act, 10))
