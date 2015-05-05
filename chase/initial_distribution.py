import numpy as np


def uniform_initial_distribution(N, *args):
    """For a given number of states N, return a
    uniform distribution."""
    Z = np.matrix(np.ones(N) / N)
    return Z


def softmax_initial_distribution(N, pars):
    tau = pars.get('tau', 1.)
    raise NotImplementedError
