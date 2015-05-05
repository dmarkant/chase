


def uniform_initial_distribution(N):
    """For a given number of states N, return a
    uniform distribution."""
    Z = np.matrix(np.ones(N) / N)


def softmax_initial_distribution(N, tau=1.):
    raise NotImplementedError
