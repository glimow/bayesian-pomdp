import math

__author__ = 'philippe'


def num_integral(values):
    """
    Integral approximation using a sum of triangles.
    :param values: uniformly sampled trajectory values (x,y).
    :return: approximation of the trajectory norm.
    """
    d = 0
    for i in range(len(values) - 1):
        d += math.sqrt((values[i + 1][0] - values[i][0]) ** 2 + (values[i + 1][1] - values[i][1]) ** 2)
    return d


def UCB(k):
    """
    Upper confidence bound function used in Bayesian Optimization
    :param k: exploration-exploitation parameter
    :return: UCB function of two values (mean and std), with parameter k
    """

    def fun(mean, std):
        return mean + k * std

    return fun
