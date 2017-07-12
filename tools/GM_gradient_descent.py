import numpy as np

__author__ = 'philippe'

def gm_gradient_descent(_pis, _mus, _covs, D, max_it=100):
    """
    Runs several gradient descent algorithms to compute the modes of a Gaussian Mixture.
    :param _pis: List of GM coefficients.
    :param _mus: List of means.
    :param _covs: List of covariance matrices.
    :param D: problem dimension.
    :param max_it: Number of gradient descent iteration for each GM element.
    :return: 2 arrays:
             - An array with the modes of the given GM.
             - An array with the GM values at the modes.
    """
    assert len(_pis) == len(_mus) == len(_covs)
    m = len(_pis)

    # Precompute
    prob_facs = [pi * np.power(2.0 * np.pi * np.linalg.det(cov), -0.5) for cov, pi in zip(_covs, _pis)]
    inv_covs = [np.linalg.inv(cov) for cov in _covs]

    def pxms(_x):
        mean_ds = _x - _mus
        return np.array([f * np.exp(-0.5 * np.dot(np.dot(d.T, c), d)) for f, d, c in zip(prob_facs, mean_ds, inv_covs)])

    def prob(_x):
        return np.sum(pxms(_x))

    def grad_ln(px, _x):
        mean_ds = _mus - _x
        pxmss = pxms(_x)
        _g = sum(pxm * np.dot(c, d) for pxm, c, d in zip(pxmss, inv_covs, mean_ds))
        return np.atleast_2d(_g / px)

    def hessian_ln(px, _g, _x):
        mean_ds = _mus - _x
        _H = sum(
            pxm * np.dot(np.dot(ic, np.dot(d, d.T) - c), ic) for pxm, ic, c, d in
            zip(pxms(_x), inv_covs, _covs, mean_ds))
        a = -np.dot(_g.T, _g)
        return a + _H / px

    # Constants
    sigma = np.sqrt(min(min(np.linalg.eigvals(cov)) for cov in _covs))
    epsilon = 10e-4

    # Control parameters
    pisig = (2.0 * np.pi * sigma ** 2) ** (D / 2.0)
    min_step = sigma ** 2 * pisig
    min_grad = epsilon * np.exp(-0.5 * epsilon ** 2) / (pisig * sigma)
    min_diff = 1000.0 * epsilon * sigma
    max_eig = 0

    # Initialize
    s = 64 * min_step
    modes = []

    # Optionally TODO: ?
    # Could remove all components for which their coefficient is < theta=10e-2 and renormalize pm

    # Core loop
    for m in range(0, m):
        i = 0
        x = _mus[m]
        p = prob(x)
        g = np.array([min_grad, min_grad])  # Something to get in the while loop
        H = []
        while i < max_it and np.linalg.norm(g, ord=2) >= min_grad:
            g = grad_ln(p, x)
            H = hessian_ln(p, g, x)
            x_old = x
            p_old = p
            H_neg = np.all(np.linalg.eigvals(H) < 0)  # Is H negative definite?
            if H_neg:
                x = x_old - np.dot(np.linalg.inv(H), g.T).T
                p = prob(x)
            if not H_neg or p <= p_old:
                x = x_old + s * g
                p = prob(x)
                while p < p_old:
                    s /= 2.0
                    x = x_old + s * g
                    p = prob(x)
            i += 1
        ns = [v for v in modes if np.linalg.norm(v - x) <= min_diff]
        ns.append(x.reshape(-1).tolist())
        modes = [mode for mode in modes if mode not in ns]
        modes.append(ns[np.argmax(prob(v) for v in ns)])

    return np.array(modes), np.array([prob(mode) for mode in modes])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    n_elem = 20

    pis = np.random.uniform(0.0, 1.0, (n_elem,))
    mus = np.random.uniform(0.0, 1.0, (n_elem, 2))
    covs = np.array([np.identity(2) * np.random.uniform(0.005, 0.1) for _ in range(n_elem)])

    all_modes, all_values = gm_gradient_descent(pis, mus, covs, 2, max_it=100)
    print 'Modes found:', all_modes
    print 'And values:', all_values

    res = 50
    axis_x = [0, 1]
    axis_y = [0, 1]
    space_x = np.linspace(axis_x[0], axis_x[1], res)
    space_y = np.linspace(axis_y[0], axis_y[1], res)
    xv, yv = np.meshgrid(space_x, space_y)
    space_xy = np.array([xv.reshape(-1), yv.reshape(-1)]).T

    _prob_facs = [_pi * np.power(2.0 * np.pi * np.linalg.det(_cov), -0.5) for _cov, _pi in zip(covs, pis)]
    _inv_covs = [np.linalg.inv(_cov) for _cov in covs]


    def est(x):
        _mean_ds = x - mus
        return sum(
            np.array(
                [f * np.exp(-0.5 * np.dot(np.dot(d.T, c), d)) for f, d, c in zip(_prob_facs, _mean_ds, _inv_covs)]))


    mixt_g = np.array([est(xy) for xy in space_xy]).reshape((res, res))
    plt.figure(figsize=(14, 10))
    plt.subplot(1, 1, 1)
    plt.contourf(space_x, space_y, mixt_g, 128, alpha=.75, cmap='jet')
    plt.plot(mus[:, 0], mus[:, 1], 'r*')
    if len(all_modes) > 0:
        plt.plot(all_modes[:, 0], all_modes[:, 1], 'bo')
    plt.show()
