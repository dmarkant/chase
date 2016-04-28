import numpy as np
import cpt
from time import time

class DriftModel(object):
    """EV drift model.

    """
    def __init__(self, **kwargs):
        self.problems = kwargs.get('problems', None)

    def __call__(self, options, pars={}, **kwargs):
        """Evaluate the drift rate for a given
        set of parameters"""

        c = pars.get('c', .5)
        se = self.evaluation(options, pars)
        V = se['V']
        sigma2 = se['sigma2']
        d = c * (V[1] - V[0]) / (np.sqrt(sigma2))

        assert not np.isnan(d)
        d = self.truncate(d)
        return d


    def truncate(self, d):
        # drift must be bounded by -1, 1
        if d <= -1:
            d = -.99999
        elif d >= 1:
            d = .99999
        return d


    def evaluation(self, options, pars):

        values = options[:,:,0]
        weights = options[:,:,1]

        # expected value of each outcome
        v = np.array([np.multiply(weights[i], values[i]) for i in range(len(options))])

        # expected variance of each option
        evar = np.array([np.dot(weights[i], values[i] ** 2) - np.sum(v[i]) ** 2 for i in range(len(options))])

        # expected value of each option
        V = np.sum(v, axis=1)

        # combine variances (ensure > 0)
        sigma2 = np.max([np.sum(evar), 1e-10])

        return {'weights': weights,
                'values': values,
                'evar': evar,
                'V': V,
                'sigma2': sigma2}


class CPTDriftModel(DriftModel):
    """CPT drift model."""

    def __init__(self, **kwargs):
        super(CPTDriftModel, self).__init__()
        self.problems = kwargs.get('problems', None)

        if self.problems is not None:
            self.rdw = cpt.setup(self.problems)
        else:
            self.rdw = None


    def evaluation(self, options, pars):

        if self.rdw is None or 'probid' not in pars:
            weights = np.array([cpt.pweight_prelec(option, pars) for i, option in enumerate(options)])
        else:
            probid = pars['probid']
            weights = np.array([cpt.pweight_prelec_known_problem(self.rdw[probid][i], pars) for i, option in enumerate(options)])

        values = np.array([cpt.value_fnc(option[:,0], pars) for option in options])

        # expected value of each outcome
        v = np.array([np.multiply(weights[i], values[i]) for i in range(len(options))])

        # expected variance of each option
        evar = np.array([np.dot(weights[i], values[i] ** 2) - np.sum(v[i]) ** 2 for i in range(len(options))])

        # expected value of each option
        V = np.sum(v, axis=1)

        # combine variances (ensure > 0)
        sigma2 = np.max([np.sum(evar), 1e-10])

        try:
            assert not np.any(np.isnan(evar))
        except AssertionError:
            print weights
            print evar
            print dummy

        return {'weights': weights,
                'values': values,
                'evar': evar,
                'V': V,
                'sigma2': sigma2}



class SwitchingCPTDriftModel(CPTDriftModel):
    """CPT drift model with switching between options"""

    def __init__(self, **kwargs):
        super(SwitchingCPTDriftModel, self).__init__(**kwargs)


    def __call__(self, sampled, options, pars={}, **kwargs):
        """Evaluate the drift rate for a given
        set of parameters"""

        c = pars.get('c', .5)
        se = self.evaluation(options, pars)
        V = se['V']
        evar = se['evar']

        if sampled==0:  d = c * (-V[0]) / (np.sqrt(np.max([evar[0], 1e-10])))
        else:           d = c * (V[1]) / (np.sqrt(np.max([evar[1], 1e-10])))

        d = self.truncate(d)
        return d


if __name__ == '__main__':

    options = [[[  4.        ,   0.5       ],
                [ 12.        ,   0.26315789],
                [  0.        ,   0.23684211]],
               [[  1.  ,   0.41],
                [  0.  ,   0.28],
                [ 18.  ,   0.31]]]
    options = np.array(options)

    m = CPTDriftModel()

    pars = {'prelec_gamma': 1.5}

    print m(options, pars=pars)

