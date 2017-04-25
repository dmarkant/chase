import numpy as np


def indifferent_initial_distribution(N, *args):
    """1 at the center state, 0 elsewhere"""
    assert (N % 2 == 1), "The number of states has to be odd!"
    Z = np.matrix(np.zeros(N))
    Z[0,(N-1)/2] = 1.
    return Z


def uniform_initial_distribution(N, *args):
    """For a given number of states N, return a
    uniform distribution."""
    assert (N % 2 == 1), "The number of states has to be odd!"
    Z = np.matrix(np.ones(N) / N)
    return Z


def softmax_initial_distribution(N, pars):
    """Softmax function over starting points

    tau: inverse temperature (low -> uniform)
    """
    assert (N % 2 == 1), "The number of states has to be odd!"
    tau = pars.get('tau', 1.)

    V = np.arange(N) - (N - 1)/2
    Z = np.exp(-np.abs(V) * (float(tau)))
    Z = np.matrix(Z / np.sum(Z))
    return Z


def laplace_initial_distribution(N, pars):
    """Laplace distribution over starting points"""
    assert (N % 2 == 1), "The number of states has to be odd!"
    p = pars.get('tau', .5)
    p = np.clip(p, 0, 1 - 1e-10)
    S = np.arange(N) - (N - 1)/2
    theta = pars.get('theta')
    assert (theta == (N-1) / 2), "Number of states does not follow from theta"

    def F(x):
        if x < 0:
            return (p ** (- np.floor(x))) / (1 + p)
        else:
            return 1 - (p ** (np.floor(x) + 1))/(1 + p)

    num = ((1 - p)/(1+p)) * (p ** np.abs(S))
    den = F(theta - 1) - F(-(theta))
    return np.matrix(num / den)


if __name__ == '__main__':

    #d = softmax_initial_distribution(9, {})
    #print d
    #print d.sum()

    d = laplace_initial_distribution(61, {'tau': .7089, 'theta': 31})
    print d
    print d.sum()
