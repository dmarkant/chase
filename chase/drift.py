import numpy as np
import cogmod.cpt as cpt


class DriftModel(object):
    """EV drift model.

    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, options, pars={}, **kwargs):
        """Evaluate the drift rate for a given
        set of parameters"""
        c = kwargs.get('c', .5)
        gamma = kwargs.get('gamma', 0.)
        state = kwargs.get('state', None)
        sdw = gamma * state # state-dependent weighting

        se = self.evaluation(options, pars)
        V = se['V']
        sigma2 = se['sigma2']
        d = c * (V[1] - V[0] + sdw) / (np.sqrt(sigma2))

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
                'V': V,
                'sigma2': sigma2}



class CPTDriftModel(DriftModel):

    """CPT drift model."""

    def __init__(self, **kwargs):
        super(CPTDriftModel, self).__init__()


    def evaluation(self, options, pars):

        # decision weights
        #weights = np.array([cpt.rank_dependent_weights(option, pars) for i, option in enumerate(options)])
        weights = np.array([cpt.pweight_prelec(option, pars) for i, option in enumerate(options)])

        # value weights
        values = np.array([cpt.value_fnc(option[:,0], pars) for option in options])

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
                'V': V,
                'sigma2': sigma2}



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

