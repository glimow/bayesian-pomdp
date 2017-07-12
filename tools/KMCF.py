import numpy as np
from cvxopt import matrix, solvers
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances

__author__ = 'philippe'

class KMCF:
    # Kernel on the states (squared exp on the distance between x and y)
    def ker_s(self, _x, _y):
        return np.exp(-self.param_ker_s * euclidean_distances(_x, _y, squared=True))

    # Kernel on the observations
    def ker_o(self, _x, _y):
        return np.exp(-self.param_ker_o * euclidean_distances(_x, _y, squared=True))

    def __init__(self, param_ker_s=0.2, param_ker_o=15.0, epsilon=0.1, delta=0.00001):
        # Regularisation parameters
        self.epsilon = epsilon
        self.delta = delta
        # Kernel parameters
        self.param_ker_s = param_ker_s
        self.param_ker_o = param_ker_o

        self.d = 0
        self.w = None
        self.Go = None
        self.Gs = None
        self.times = None
        self.pts = None

    def set_anchor_pts(self, _pts, _times, nb_tough_anchors=0, restrict_angle=None):
        # nb_tough_anchors is the number of "tough anchors", which are the first points that restrict
        # the robot's starting angle
        assert len(_pts) == len(_times)
        assert nb_tough_anchors < len(_pts)

        if restrict_angle:
            nb_tough_anchors += 1
        self.times = np.atleast_2d(_times)
        self.pts = _pts

        self.d = len(self.pts)

        # Space Kernel matrix
        self.Gs = self.ker_s(self.times, self.times)
        self.Go = self.ker_o(self.pts, self.pts)

        # Empirical prior mean
        m = np.ones(self.d) / self.d

        # Building regularisation matrixes with zeroes for the first few points and Identity(d) for the rest
        if nb_tough_anchors != 0:
            # Building regularisation matrixes with zeroes for the first few points and Identity(d) for the rest
            reg_mat_o = self.delta * np.pad(np.identity(self.d - nb_tough_anchors),
                                            pad_width=((nb_tough_anchors, 0), (nb_tough_anchors, 0)),
                                            mode='constant', constant_values=0.0000001)
            reg_mat_s = self.epsilon * np.pad(np.identity(self.d - nb_tough_anchors),
                                              pad_width=((nb_tough_anchors, 0), (nb_tough_anchors, 0)),
                                              mode='constant', constant_values=0.0000001)
        else:
            reg_mat_o = self.delta * np.identity(self.d)
            reg_mat_s = self.epsilon * np.identity(self.d)

        # KMCF
        lamb = np.diag(np.dot(np.linalg.inv(self.Gs + self.d * reg_mat_s), m))
        lg = np.dot(lamb, self.Go)
        inv = np.linalg.inv(np.linalg.matrix_power(lg, 2) + reg_mat_o)
        self.w = np.dot(np.dot(lg, inv), lamb)

    def test_mean(self, test_times):
        assert self.Go is not None and self.Gs is not None and self.pts is not None and self.times is not None and \
               self.w is not None and self.d > 0

        ky = self.ker_o(self.times, test_times)
        w = np.dot(self.w, ky).T

        # Normalize each line of weights
        row_sums = w.sum(axis=1)
        w_nrm = w / row_sums[:, np.newaxis]

        # Compute trajectory coordinates
        return np.dot(w_nrm, self.pts)

    def test_mix_gaussian(self, test_times):
        """
        Returns the weights of a mixture of gaussian for each test time.
        :param test_time:
        :return:
        """
        assert self.Go is not None and self.Gs is not None and self.pts is not None and self.times is not None and \
               self.w is not None and self.d > 0

        ky = self.ker_o(self.times, test_times)
        w = np.dot(self.w, ky).T

        # Normalize each line of weights
        row_sums = w.sum(axis=1)
        w_nrm = w / row_sums[:, np.newaxis]

        # Compute means
        means = self.test_mean(self.times)

        # Compute the mixture of gaussian
        k = 1.0
        # A_ = np.power(2 * np.pi, k / 2) * rbf_kernel(self.pts, self.pts, gamma=self.param_ker_s)
        # B_ = np.power(2 * np.pi, k / 2) * rbf_kernel(self.pts, self.pts, gamma=self.param_ker_s)
        A_ = np.power(2 * np.pi, k / 2) * rbf_kernel(means, means, gamma=self.param_ker_s)
        B_ = np.power(2 * np.pi, k / 2) * rbf_kernel(means, self.pts, gamma=self.param_ker_s)
        lam = 1.0

        mixt_w = np.ndarray(w_nrm.shape)
        P = matrix(A_ + np.identity(len(A_)) * lam)

        # Matrices for x >= 0
        G = matrix(-np.identity(w_nrm.shape[1]))
        h = matrix([0.0] * w_nrm.shape[1])

        # Matrices for 1.T . x = 1
        A = matrix([1.0] * w_nrm.shape[1], (1, w_nrm.shape[1]))
        b = matrix(1.0)

        solvers.options['show_progress'] = False
        for i in range(w_nrm.shape[0]):
            q = matrix(-np.dot(B_.T, w_nrm[i]))
            sol = solvers.qp(P, q, G, h, A, b)
            mixt_w[i] = np.array(sol['x']).reshape(-1)

        def mixt_ker(x):
            return rbf_kernel(x, self.pts, gamma=30)

        return mixt_w, mixt_ker


if __name__ == '__main__':
    kmcf = KMCF()
    pts = np.array([[0.1, 0.9, 0.5, 0.5, 0.5, 0.1, 0.9]]).T
    times = np.array([[0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 0.9]]).T
    kmcf.set_anchor_pts(pts, times)
    print kmcf.test_mean(np.linspace(0, 1, 50)[:, np.newaxis])
