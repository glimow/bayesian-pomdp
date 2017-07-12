import random
from environment.Splines2D import DiscreteSplines2D

__author__ = 'philippe'


class RandomBehavior:
    def __init__(self):
        pass

    def run(self, from_pos, belief):
        """
        Runs the behavior
        :param from_pos: Pose from which to start planning.
        :param belief: dummy argument.
        :return: a random spline action.
        """
        splines = DiscreteSplines2D.get_splines(from_pos)
        return splines[random.randint(0, len(splines) - 1)]
