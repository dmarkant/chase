import numpy as np
from scipy.stats import norm, geom
"""Alternate stopping rules"""


class TruncatedNormal(object):

    def dist(self, pars):

        mu = pars.get('mu', 10)
        sc = pars.get('sc', 5)
        t1 = np.arange(1, 300)
        t0 = t1 - 1

        p = norm.cdf(t1, loc=mu, scale=sc) - norm.cdf(t0, loc=mu, scale=sc)
        p = p / (norm.cdf(np.inf, loc=mu, scale=sc) - norm.cdf(0, loc=mu, scale=sc))
        return p


class Geometric(object):

    def dist(self, pars):
        t = np.arange(1, 300)
        p = pars.get('p_stop', .1)
        pr = geom.pmf(t, p)
        return pr


if __name__=='__main__':

    test = Geometric()
    p = test.dist({'p_stop': .2})

    print p.sum()
    print p
