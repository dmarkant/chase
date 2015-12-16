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


def laplace_initial_distribution(N, pars):
    p = pars.get('tau', .5)
    S = np.arange(N) - (N - 1)/2
    theta = pars.get('theta')

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
