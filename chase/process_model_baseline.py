from __future__ import division
import numpy as np
import cpt
from copy import deepcopy
from time import time
from utils import *
from initial_distribution import laplace_initial_distribution
from scipy.stats import geom


class CHASEBaselineProcessModel(object):

    def __init__(self, **kwargs):
        pass

    def __call__(self, options, pars):
        """Simulate process model to get predicted
        choice and sample size distributions"""

        start = time()

        N = pars.get('N', 500000)
        max_T = pars.get('max_T', 500)
        minsamplesize = pars.get('minsamplesize', 1) - 1
        p_stop_geom = pars.get('p_stop_geom', 0)
        fixed_dist = geom(p_stop_geom, loc=(minsamplesize - 1))

        outcomes = pars['obs']['outcomes']
        samplesize = outcomes.shape[0]
        fixed_p_stop = fixed_dist.pmf(samplesize - 1)

        p_stop_choose_A = fixed_p_stop * 0.5
        p_stop_choose_B = fixed_p_stop * 0.5

        return {'p_stop_choose_A': p_stop_choose_A,
                'p_stop_choose_B': p_stop_choose_B}


    def nloglik(self, problems, data, pars):
        """For a single set of parameters, evaluate the
        log-likelihood of observed data set."""

        llh = 0

        for obs in data:

            grp = obs['group']
            pars['probid'] = obs['probid']

            # check if there are any parameters specific to this group
            grppars = {}

            nonspec = filter(lambda k: k.count('(')==0, pars)
            for k in nonspec:
                grppars[k] = pars[k]

            grp_k = filter(lambda k: k.count('(%s)' % grp)==1, pars)
            for k in grp_k:
                grppars[k.rstrip('(%s)' % grp)] = pars[k]

            grppars['obs'] = obs

            # run the model
            results = self.__call__(problems[obs['probid']], grppars)

            if obs['choice']==0:
                llh += np.log(pfix(results['p_stop_choose_A']))
            elif obs['choice']==1:
                llh += np.log(pfix(results['p_stop_choose_B']))
            else:
                print obs['choice']
                print dummy

        return -llh


    def nloglik_opt(self, value, problems, data, pars):
        pars, fitting, verbose = unpack(value, pars)

        start = time()

        # check if the parameters are within allowed range
        if outside_bounds(value, fitting):
            return np.inf
        else:
            for v, p in zip(value, fitting.keys()):
                pars[p] = v
            nllh = self.nloglik(problems, data, pars)
            print np.round(value, 3), nllh, time()-start
            return nllh


if __name__=='__main__':


    # options are two normal distributions, each [m, sd]
    #problems = np.array([[[3, 1.], [1, 1.]]])

    #simulate_process(problems)

    problems = {}
    arr = np.genfromtxt('paper/data/six_problems.csv', delimiter=',')
    for i in range(len(arr)):
        problems[i] = arr[i].reshape((2,2,2))


    m = CHASEProcessModel(problems=problems)

    for pid in problems:
        pars = {'probid': pid,
                'p_stay': .1,
                'prelec_gamma': 1}
        m(problems[pid], pars)

