__author__ = 'philippe'


class Pose:
    def __init__(self, x, y, z=0, p=0, r=0, w=0, w_vel=0, t=0):
        """
        Initializing a Pose.
        :param x: x
        :param y: y
        :param z: z (default 0)
        :param p: pitch (default 0)
        :param r: raw (default 0)
        :param w: yaw (default 0)
        :param w_vel: yaw velocity (default 0)
        :param t: time (default 0)
        """
        self.x = x
        self.y = y
        self.z = z
        self.p = p
        self.r = r
        self.w = w
        self.w_vel = w_vel
        self.t = t

    def clone(self):
        """
        Returns a copy of the current pose.
        :return: a copy of the current pose.
        """
        return Pose(self.x, self.y, self.z, self.p, self.r, self.w, self.t)

    @staticmethod
    def to_xyw_array(pos):
        """
        Returns a [x, y, yaw] list for the given pose.
        :param pos: a Pose
        :return: a [x, y, yaw] list
        """
        return [pos.x, pos.y, pos.w]

    @staticmethod
    def to_xy_array(pos):
        """
        Returns a [x, y] list for the given pose.
        :param pos: a Pose
        :return: a [x, y] list
        """
        return [pos.x, pos.y]

    @staticmethod
    def to_xyt_array(pos):
        """
        Returns a [x, y, t] list for the given pose.
        :param pos: a Pose
        :return: a [x, y, t] list
        """
        return [pos.x, pos.y, pos.t]

    @staticmethod
    def to_xyz_array(pos):
        """
        Returns a [x, y, z] list for the given pose.
        :param pos: a Pose
        :return: a [x, y, z] list
        """
        return [pos.x, pos.y, pos.z]

    def __repr__(self):
        return '{}\n'.format(self.__str__())

    def __str__(self):
        """
        String representation of the pose.
        :return: a String representation of the pose.
        """
        return '* x:{:6.3f}, y:{:6.3f}, z:{:6.3f}\n' \
               '  p:{:6.3f}, r:{:6.3f} ,w:{:6.3f}\n' \
               '  t:{:6.3f}, w_vel:{:6.3f}'.format(self.x, self.y, self.z, self.p, self.r, self.w, self.t, self.w_vel)
