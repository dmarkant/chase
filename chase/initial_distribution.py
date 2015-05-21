import numpy as np


def indifferent_initial_distribution(N, *args):
    """1 at the center state, 0 elsewhere"""
    Z = np.matrix(np.zeros(N))
    Z[0,(N-1)/2.] = 1.
    return Z


def uniform_initial_distribution(N, *args):
    """For a given number of states N, return a
    uniform distribution."""
    Z = np.matrix(np.ones(N) / N)
    return Z


def softmax_initial_distribution(N, pars):
    tau = pars.get('tau', 1.)

    V = np.arange(N) - (N - 1)/2
    Z = np.exp(-np.abs(V) * (float(tau)))
    Z = np.matrix(Z / np.sum(Z))
    return Z
