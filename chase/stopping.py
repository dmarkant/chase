"""Alternate stopping rules"""


class TruncatedNormal(object):

    def __init__(self):
        pass


    def nloglik(self, pars):

        mu = pars.get('mu', 10)
        sc = pars.get('sc', 5)
        t1 = np.arange(1, 300)
        t0 = t1 - 1

        p = norm.cdf(t1, loc=mu, scale=sc) - norm.cdf(t0, loc=mu, scale=sc)
        p = p / (norm.cdf(np.inf, loc=mu, scale=sc) - norm.cdf(0, loc=mu, scale=sc))
        return p
