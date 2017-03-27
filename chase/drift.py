import numpy as np
import cpt


class DriftModel(object):
    """EV drift model.

    """
    def __init__(self, **kwargs):
        self.problems = kwargs.get('problems', None)
        self.problemtype = kwargs.get('problemtype', 'multinomial')


    def __call__(self, options, pars={}, **kwargs):
        """Evaluate the drift rate for a given
        set of parameters.

        pref_units: Units of preference change ('sums' | 'diffs')
        sc: Drift scaling factor (applied to all problems)
        p_sample_H: probability of sampling the H option
        """
        optiontype = pars.get('optiontype', 'multinomial')
        pref_units = pars.get('pref_units', 'diffs')
        sc = pars.get('sc', 1)
        p_sample_H = pars.get('p_sample_H', .5)

        # subjective evaluation of options
        se = self.evaluate(options, pars)
        V = se['V'] # expected values
        evar = se['evar']
        sigma2 = se['sigma2'] # expected variances

        if pref_units == 'diffs':
            # mean change is difference in option valences
            delta = V[1] - V[0]

        elif pref_units == 'sums':
            # mean change is weighed sum of option valences
            delta = (p_sample_H * V[1] - (1 - p_sample_H) * V[0])

        # drift
        d = delta / (sc * np.sqrt(sigma2))

        try:
            assert not np.isnan(d)
        except:
            print d
            print dummy

        # drift must be bounded by -1 and 1
        d = np.clip(d, -.99999, .99999)

        if optiontype is 'multinomial':
            return se
        else:
            return d


    def weighting(self, options, pars):
        optiontype = pars.get('optiontype', 'multinomial')

        if optiontype is 'normal':
            weights = None
            values = None
            V = options[:,0]
            evar = options[:,1]
        elif optiontype is 'multinomial':
            values = options[:,:,0]
            weights = options[:,:,1]

            # weighted value of each outcome
            v = np.array([np.multiply(weights[i], values[i]) for i in range(len(options))])

            # expected value of each option
            V = np.sum(v, axis=1)
            evar = None

        return weights, values, V, evar


    def evaluate(self, options, pars):

        optiontype = pars.get('optiontype', 'multinomial')
        pref_units  = pars.get('pref_units', 'diffs')
        p_sample_H = pars.get('p_sample_H', .5)
        p_sample_L = 1 - p_sample_H
        p_stay = pars.get('p_stay', 0)

        t = pars.get('t', None)


        weights, values, V, evar = self.weighting(options, pars)

        if optiontype is 'normal':

            if pref_units == 'diffs':
                # nothing else to do
                pass

            elif pref_units == 'sums':

                # mean difference
                #delta = (1-p_stay) * (c * V[1] - (1 - c) * V[0])
                delta = (p_sample_H * V[1] - (1 - p_sample_H) * V[0])

                # variance around the mean preference change
                sigma0 = evar[0] + (-V[0] - delta)**2
                sigma1 = evar[1] + ( V[1] - delta)**2
                evar = np.array([sigma0, sigma1])

        elif optiontype is 'multinomial':

            if pref_units == 'diffs':

                # expected variance of each option
                evar = np.array([np.dot(weights[i], values[i] ** 2) - V[i] ** 2 for i in range(len(options))])

            elif pref_units == 'sums':

                # expected change in preference
                #delta = (1-p_stay) * (p_sample_H * V[1] - (1 - p_sample_H) * V[0])
                delta = (p_sample_H * V[1] - p_sample_L * V[0])

                # expected variance of valences
                evar = np.array([np.dot(weights[i], values[i] ** 2) - delta ** 2 for i in range(len(options))])

        # for both diffs and sums, the expected variance in
        # preference change is weighted sum from sampling each option
        sigma2 = p_sample_H * evar[1] + p_sample_L * evar[0]


        if t is not None and pref_units == 'diffs':

            # after t + 1 trials, what is the expected variance of the mean
            # for each option?
            trial = t + 1
            sigma2 = p_sample_H * (evar[1] + evar[0]/trial) + p_sample_L * (evar[0] + evar[1]/trial)


        # truncate close to zero
        sigma2 = np.clip(sigma2, 1e-10, np.inf)

        return {'weights': weights,
                'values': values,
                'evar': evar,
                'V': V,
                'sigma2': sigma2,
                'd_down': d_down,
                'd_up': d_up}



class CPTDriftModel(DriftModel):
    """CPT drift model."""

    def __init__(self, **kwargs):
        super(CPTDriftModel, self).__init__(**kwargs)

        # setup rank-dependent sorting of options to speed up fitting later
        if self.problemtype is 'multinomial' and self.problems is not None:
            self.rdw = cpt.setup(self.problems)
        else:
            self.rdw = None


    def weighting(self, options, pars):
        optiontype = pars.get('optiontype', 'multinomial')

        if optiontype is 'multinomial':
            values = options[:,:,0]
            weights = options[:,:,1]

            # decision weighting
            if 'prelec_gamma' in pars or 'prelec_elevation' in pars:
                if self.rdw is None: wopt = options
                else:                wopt = self.rdw[pars['probid']]
                weights = np.array([cpt.pweight_prelec(option, pars) for option in wopt])

            # value weighting
            if 'pow_gain' in pars or 'w_loss' in pars:
                values = np.array([cpt.value_fnc(option[:,0], pars) for option in options])

            # expected value of each outcome
            v = np.array([np.multiply(weights[i], values[i]) for i in range(len(options))])

            # expected value of each option
            V = np.sum(v, axis=1)
            evar = None

        elif optiontype is 'normal':
            weights = None
            values = None
            V = options[:,0]
            evar = options[:,1]

            if 'pow_gain' in pars:

                # calculated expected value and variance given
                # outcome transformation
                ev0, evar0 = cpt.normal_raised_to_power(options[0], pars['pow_gain'])
                ev1, evar1 = cpt.normal_raised_to_power(options[1], pars['pow_gain'])

                V = np.array([ev0, ev1])
                evar = np.array([evar0, evar1])

        return weights, values, V, evar



class SwitchingCPTDriftModel(CPTDriftModel):
    """CPT drift model with switching between options"""

    def __init__(self, **kwargs):
        super(SwitchingCPTDriftModel, self).__init__(**kwargs)


    def __call__(self, sampled, options, pars={}, **kwargs):
        """Evaluate the drift rate for a given
        set of parameters"""

        se = self.evaluate(options, pars)
        V = se['V']
        evar = se['evar']
        sigma2 = np.max([np.sum(evar), 1e-10])

        #n = 1
        #if sampled==0:  d = -(V[0])/(n + np.sqrt(np.max([evar[0], 1e-10])))
        #else:           d =  (V[1])/(n + np.sqrt(np.max([evar[1], 1e-10])))

        n = 0
        if sampled==0:  d = -(V[0])/(n + np.sqrt(sigma2))
        else:           d =  (V[1])/(n + np.sqrt(sigma2))

        #if sampled==0:  d = (.25 * -V[0]) / (np.sqrt(np.max([evar[0], 1e-10])))
        #else:           d =  (.25 * V[1]) / (np.sqrt(np.max([evar[1], 1e-10])))

        d = np.clip(d, -.99999, .99999)
        return d


if __name__ == '__main__':

    options = [[[  4.  ,   0.5 ],
                [ 12.  ,   0.26],
                [  0.  ,   0.23]],
               [[  1.  ,   0.41],
                [  0.  ,   0.28],
                [ 18.  ,   0.31]]]
    options = np.array(options)

    m = CPTDriftModel()

    pars = {'prelec_gamma': 1.5}

    print m(options, pars=pars)

