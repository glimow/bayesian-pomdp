import copy
import math

from environment.state import State
from tools import putils

__author__ = 'philippe'


class Splines2D:
    """
    Class used to generate Cubic Splines.
    """
    free_dims = 4
    norm_spline = 0.6
    splines_scale = 1

    def __init__(self):
        pass

    @staticmethod
    def get_spline_lowdim(spl):
        """
        Returns the low dimension representation of a Spline.
        :param spl: Spline from which to get the low dimension representation.
        :return: a list of low dimension parameters.
        """
        return spl.ld_params

    @staticmethod
    def gen_spline_from_lowdim(ld_params, pos):
        """
        Generate a Cubic Spline from low dimension parameters.
        :param ld_params: a list of low dimension parameters of size 4 or 5.
        :param pos: state to constrain the Spline to.
        :return: a generated Spline.
        """
        # Create a new spline
        by = math.sin(pos.w_vel) * ld_params[2]
        if Splines2D.free_dims == 5:
            by = ld_params[4]
        spl = Spline2D(ld_params[0],
                       math.cos(pos.w_vel) * ld_params[2],
                       math.cos(pos.w) * ld_params[3],
                       pos.x,
                       ld_params[1],
                       by,
                       math.sin(pos.w) * ld_params[3],
                       pos.y,
                       ld_params, Splines2D.splines_scale)

        # Normalize the spline
        integral_steps = 30
        steps = [du / float(integral_steps) for du in range(0, integral_steps + 1)]
        nrm = putils.num_integral(map(lambda u: state.to_xy_array(spl.pos_at(u)), steps)) / Splines2D.norm_spline
        if nrm > 0.001:
            spl.ax /= nrm
            spl.ay /= nrm
            spl.bx /= nrm
            spl.by /= nrm
            spl.cx /= nrm
            spl.cy /= nrm

        return spl


class Spline2D:
    def __init__(self, ax, bx, cx, dx, ay, by, cy, dy, ld_params=None, scale=1):
        """
        Cubic Spline initialization. Parameters fit the following equations:
        x(t) = ax * t^3 + bx * t^2 + cx * t + dx
        y(t) = ay * t^3 + by * t^2 + cy * t + dy
        :param ax: ax
        :param bx: bx
        :param cx: cx
        :param dx: dx
        :param ay: ay
        :param by: by
        :param cy: cy
        :param dy: dy
        :param ld_params: Low dimension parameters (used to build a spline in the continuous case). (default None)
        :param scale: scaling factor to apply to the spline parameters. (default 1)
        """
        self.ld_params = ld_params
        self.scale = scale
        self.ax = ax * scale
        self.bx = bx * scale
        self.cx = cx * scale
        self.dx = dx * scale
        self.ay = ay * scale
        self.by = by * scale
        self.cy = cy * scale
        self.dy = dy * scale

    def pos_at(self, u):
        """
        Get a position by sampling the spline at a specific time.
        :param u: Time at which to sample.
        :return: a state.
        """
        _x = self.cx * u + self.dx
        _y = self.ay * math.pow(u, 3) + self.by * math.pow(u, 2) + self.cy * u + self.dy
        der1 = (3.0 * self.ax * math.pow(u, 2) + 2.0 * self.bx * u + self.cx,
                3.0 * self.ay * math.pow(u, 2) + 2.0 * self.by * u + self.cy)
        der2 = (3.0 * self.ax * u + self.bx, 3.0 * self.ay * u + self.by)
        return State(_x, _y, w=math.atan2(der1[1], der1[0]), w_vel=math.atan2(der2[1], der2[0]))

    def __str__(self):
        """
        Print user-friendly string representation of Spline.
        :return: string representation of the Spline.
        """
        if self.ld_params is None:
            s = 'x:{}, {}, {}, {}\t y:{}, {}, {}, {}'
            return s.format(self.ax, self.bx, self.cx, self.dx, self.ay, self.by, self.cy, self.dy)
        else:
            s = 'x:{}, {}, {}, {}\t y:{}, {}, {}, {}\t\t (low dim:{}, {}, {}, {})'
            return s.format(self.ax, self.bx, self.cx, self.dx, self.ay, self.by, self.cy, self.dy, self.ld_params[0],
                            self.ld_params[1], self.ld_params[2], self.ld_params[3])


class DiscreteSplines2D:
    """
    Class used to retrieve predefined Splines.

    Definition of 10 Splines:
    """
    __default_splines = [
        # Forward
        Spline2D(0, 0, 0.72, 0, 0, 0, 0, 0),  # Straight
        Spline2D(0, 0, 0.55, 0, 0, 0.5, -0.1, 0),  # Slight left turn
        Spline2D(0, 0, 0.55, 0, 0, -0.5, 0.1, 0),  # Slight right turn
        Spline2D(0, -0.33, 0.71, 0, 0, 0.52, 0, 0),  # Harsh left turn
        Spline2D(0, -0.33, 0.71, 0, 0, -0.52, 0, 0)  # Harsh right turn
    ]
    __default_splines2 = [
        # # using 9 actions
        Spline2D(0, 0, 0.65, 0, 0, 0.34, -0.08, 0),  # Very Slight left turn
        Spline2D(0, 0, 0.65, 0, 0, -0.34, 0.08, 0),  # Very Slight right turn
        Spline2D(0, -0.20, 0.67, 0, 0, 0.475, 0, 0),  # Medium left turn
        Spline2D(0, -0.20, 0.67, 0, 0, -0.475, 0, 0)  # Medium right turn
        #
    ]
    __default_splines3 = [
        # # using 17 actions
        Spline2D(0, 0, 0.705, 0, 0, 0.14, -0.02, 0),  # Very Very Slight left turn
        Spline2D(0, 0, 0.705, 0, 0, -0.14, 0.02, 0),  # Very Very Slight right turn
        Spline2D(0, 0, 0.61, 0, 0, 0.40, -0.08, 0),  # Quite Very Slight left turn
        Spline2D(0, 0, 0.61, 0, 0, -0.40, 0.08, 0),  # Quite Very Slight right turn
        Spline2D(0, -0.15, 0.66, 0, 0, 0.44, 0, 0),  # Slight Medium left turn
        Spline2D(0, -0.15, 0.66, 0, 0, -0.44, 0, 0),  # Slight Medium right turn
        Spline2D(0, -0.27, 0.69, 0, 0, 0.5, 0, 0),  # Quite Harsh left turn
        Spline2D(0, -0.27, 0.69, 0, 0, -0.5, 0, 0)  # Quite Harsh  right turn

        # Back
        # Spline2D(0, 0, -0.745, 0, 0, 0, 0, 0),  # Straight
        # Spline2D(0, 0, -0.55, 0, 0, 0.5, -0.1, 0),  # Slight right turn
        # Spline2D(0, 0, -0.55, 0, 0, -0.5, 0.1, 0),  # Slight left turn
        # Spline2D(0, 0.4, -0.8, 0, 0, 0.6, 0, 0),  # Harsh right turn
        # Spline2D(0, 0.4, -0.8, 0, 0, -0.6, 0, 0)  # Harsh left turn
    ]

    def __init__(self):
        pass

    @staticmethod
    def get_splines(pos):
        """
        Retrieve a list of discrete Splines, constrained to a position.
        :param pos: state to constrain the Splines to.
        :return: a list of Splines.
        """
        acts = copy.deepcopy(DiscreteSplines2D.__default_splines)
        for act in acts:
            DiscreteSplines2D.__constrain_spline_to__(act, pos)
        return acts

    # @staticmethod
    # def get_splines2(pos):
    #     """
    #     Retrieve a list of discrete Splines, constrained to a position.
    #     :param pos: state to constrain the Splines to.
    #     :return: a list of Splines.
    #     """
    #     acts = copy.deepcopy(DiscreteSplines2D.__default_splines2)
    #     for act in acts:
    #         DiscreteSplines2D.__constrain_spline_to__(act, pos)
    #     return acts
    #
    # @staticmethod
    # def get_splines3(pos):
    #     """
    #     Retrieve a list of discrete Splines, constrained to a position.
    #     :param pos: state to constrain the Splines to.
    #     :return: a list of Splines.
    #     """
    #     acts = copy.deepcopy(DiscreteSplines2D.__default_splines3)
    #     for act in acts:
    #         DiscreteSplines2D.__constrain_spline_to__(act, pos)
    #     return acts

    @staticmethod
    def __constrain_spline_to__(spl, pos):
        """
        Used to constrain a Spline to a specific position.
        :param spl: Spline to apply constraints to.
        :param pos: state to constrain the Spline to.
        """
        # Rotate, angle theta
        theta = pos.w
        cost = math.cos(theta)
        sint = math.sin(theta)
        ax = cost * spl.ax - sint * spl.ay
        bx = cost * spl.bx - sint * spl.by
        cx = cost * spl.cx - sint * spl.cy

        ay = sint * spl.ax + cost * spl.ay
        by = sint * spl.bx + cost * spl.by
        cy = sint * spl.cx + cost * spl.cy

        spl.ax = ax
        spl.bx = bx
        spl.cx = cx
        spl.ay = ay
        spl.by = by
        spl.cy = cy

        # Translate
        spl.dx = pos.x
        spl.dy = pos.y


if __name__ == '__main__':
    def jerk(x):
        v = np.gradient(x, axis=0)
        a = np.gradient(v, axis=0)
        j = np.gradient(a, axis=0)
        return np.sum(np.linalg.norm(j, 2, axis=1))


    spl2d = DiscreteSplines2D()
    pos = State(0, 0)
    jerks = []
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure()
    spls = spl2d.get_splines(pos)
    for spl in spls:
        t = np.linspace(0, 1, 50)
        states = [spl.pos_at(u) for u in t]
        x = np.array([state.x for state in states])
        y = np.array([state.y for state in states])
        jerks.append(jerk(np.vstack((x,y)).T))
        plt.plot(x, y, 'b')

    # spls = spl2d.get_splines2(pos)
    # for spl in spls:
    #     t = np.linspace(0, 1, 50)
    #     states = [spl.pos_at(u) for u in t]
    #     x = np.array([state.x for state in states])
    #     y = np.array([state.y for state in states])
    #     jerks.append(jerk(np.vstack((x,y)).T))
    #     plt.plot(x, y, 'b--')
    #
    # spls = spl2d.get_splines3(pos)
    # for spl in spls:
    #     t = np.linspace(0, 1, 50)
    #     states = [spl.pos_at(u) for u in t]
    #     x = np.array([state.x for state in states])
    #     y = np.array([state.y for state in states])
    #     jerks.append(jerk(np.vstack((x,y)).T))
    #     plt.plot(x, y, 'b-.')

    print np.mean(jerks[1:])
    plt.show()
