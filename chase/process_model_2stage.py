from __future__ import division
import numpy as np
import cpt
from copy import deepcopy
from time import time
from utils import *
from scipy.stats import uniform, randint, geom


class TwoStageCPTModel(object):

    def __init__(self, **kwargs):

        self.stopdist = kwargs.get('stopdist', 'fixed-T')


    def __call__(self, options, pars, trackobs=True):
        """Simulate process model to get predicted
        choice and sample size distributions"""

        start = time()

        N = pars.get('N', 500)
        max_T = pars.get('max_T', 100)

        minsamplesize = pars.get('minsamplesize', 1) - 1

        p_sample_H = pars.get('p_sample_H', .5)
        p_sample_L = 1 - p_sample_H

        if self.stopdist == 'fixed-T':
            stop_T = pars.get('stop_T', 2)
            fixed_dist = randint(stop_T, stop_T+1)
        elif self.stopdist == 'geometric':
            p_stop = pars.get('p_stop', 0)
            fixed_dist = geom(p_stop, loc=(minsamplesize - 1))



        if 'obs' in pars:

            # assume a single sequence of known observations
            sampled_option = pars['obs']['sampled_option']
            outcomes = pars['obs']['outcomes']
            max_T = outcomes.shape[0]
            fixed_p_stop = fixed_dist.pmf(max_T - 1)

            opt_exp = []
            for i, opt in enumerate(options):
                opt_exp_i = []
                for j, x in enumerate(opt):
                    ind = np.where((sampled_option==i) & (outcomes==x[0]))[0]
                    n = float(len(np.where(sampled_option==i)[0]))
                    if n > 0:
                        opt_exp_i.append([x[0], len(ind)/n])
                    else:
                        opt_exp_i.append([x[0], 0])
                opt_exp_i = np.array(opt_exp_i)

                # assume single observation of zero
                if opt_exp_i[:,1].sum()==0:
                    zero = np.where(opt_exp_i[:,0]==0)[0][0]
                    opt_exp_i[zero,1] = 1
                opt_exp.append(opt_exp_i)
            opt_exp = np.array(opt_exp)

            # compute value and attentional weights for
            # each outcome
            weights = np.array([cpt.pweight_prelec(option, pars) for i, option in enumerate(opt_exp)])
            values = np.array([cpt.value_fnc(option[:,0], pars) for option in opt_exp])

            # choice function
            s = pars.get('s', 1.) # softmax temp
            vL, vH = [np.dot(w, v) for (w, v) in zip(weights, values)]
            cp = np.exp(vH * s) / (np.exp(vH * s) + np.exp(vL * s))

            p_stop_choose_A = fixed_p_stop * (1 - cp)
            p_stop_choose_B = fixed_p_stop * cp

            return {'p_stop_choose_A': p_stop_choose_A,
                    'p_stop_choose_B': p_stop_choose_B,
                    'cp_B': cp}

        else:


            values = np.array([cpt.value_fnc(option[:,0], pars) for option in options])


            # apply a fixed sample size
            samplesize = fixed_dist.rvs(size=N)
            max_T = samplesize.max()

            sampled_option = np.zeros((N, max_T), int)


            sampled_option = np.random.choice([0,1],
                                              p=[p_sample_L, p_sample_H],
                                              size=(N, max_T))

            # assume 2nd sample is from other option
            sampled_option[:,1] = np.abs(1 - sampled_option[:,0])
            sampled_A = sampled_option==0
            sampled_B = sampled_option==1


            # observation matrix
            observed = np.zeros((N, max_T))
            observed_A = np.random.choice(range(options[0].shape[0]),
                                          size=sampled_A.sum(),
                                          p=options[0][:,1])

            observed_B = np.random.choice(range(options[1].shape[0]),
                                          size=sampled_B.sum(),
                                          p=options[1][:,1])
            observed[sampled_A] = observed_A
            observed[sampled_B] = observed_B

            # outcomes experienced
            obj_outcomes = options[:,:,0]
            outcomes = np.zeros((N, max_T))
            outcomes[sampled_A] = obj_outcomes[0][observed_A]
            outcomes[sampled_B] = obj_outcomes[1][observed_B]


            # get relative frequencies
            wopt = deepcopy(options)
            wopt[:,:,0] = values

            choice = []
            for it in range(N):

                sampled_option_i = sampled_option[it,:(samplesize[it]+1)]
                outcomes_i = outcomes[it,:(samplesize[it]+1)]

                opt_exp = []
                for i, opt in enumerate(options):
                    opt_exp_i = []
                    for j, x in enumerate(opt):
                        ind = np.where((sampled_option_i==i) & (outcomes_i==x[0]))[0]
                        n = float(len(np.where(sampled_option_i==i)[0]))
                        opt_exp_i.append([x[0], len(ind)/n])
                    opt_exp.append(opt_exp_i)
                opt_exp = np.array(opt_exp)

                weights = np.array([cpt.pweight_prelec(option, pars) for i, option in enumerate(opt_exp)])
                wopt[:,:,1] = weights

                pH = cpt.choice_prob(wopt, pars)
                if np.random.random() < pH:
                    choice.append(1)
                else:
                    choice.append(0)


            choice = np.array(choice)
            p_resp = choice.mean()

            ss_A = samplesize[choice==0]
            ss_B = samplesize[choice==1]

            p_stop_A = np.bincount(ss_A, minlength=max_T)
            p_stop_A = p_stop_A/float(p_stop_A.sum())
            p_stop_B = np.bincount(ss_B, minlength=max_T)
            p_stop_B = p_stop_B/float(p_stop_B.sum())

            p_stop_cond = np.transpose([p_stop_A, p_stop_B])

            # only include data up to choice
            sampled_option = [sampled_option[i][:(samplesize[i]+1)] for i in range(samplesize.shape[0])]
            outcomes       = [outcomes[i][:(samplesize[i]+1)] for i in range(samplesize.shape[0])]
            outcome_ind    = [observed[i][:(samplesize[i]+1)] for i in range(samplesize.shape[0])]

            return {'choice': choice,
                    'samplesize': samplesize,
                    'p_resp': np.array([1-p_resp, p_resp]),
                    'p_stop_cond': p_stop_cond,
                    'sampled_option': sampled_option,
                    'outcomes': outcomes,
                    'outcome_ind': outcome_ind
                    }


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

            if 'p_switch' in grppars:
                n_switches = 0
                sampled_option = obs['sampled_option']
                for i in range(1, sampled_option.shape[0]):
                    if sampled_option[i]!=sampled_option[i-1]:
                        n_switches += 1
                llh += np.log(pfix(grppars['p_switch'])) * n_switches + \
                       np.log(pfix(1 - grppars['p_switch'])) * (sampled_option.shape[0] - 1 - n_switches)

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


def simulate_process(problems, pars={}, trace=False, relfreq=False):

    N = pars.get('N', 10000) # number of runs

    # threshold
    theta = pars.get('theta', 1) # normalized threshold

    # unbiased starting points
    #tau = pars.get('tau', 1.)

    # noise
    p_stay = pars.get('p_stay', 0.)

    # criterion
    phi = pars.get('phi', 0.)


    # output
    samplesize = {p: np.zeros(N) for p in problems}
    choices    = {p: np.zeros(N) for p in problems}

    traces = {}
    problems_exp = {}
    thresholds = {}

    for p in problems:

        problem = problems[p]
        problems_exp[p] = []
        traces[p] = []

        # weighting functions
        weights = np.array([cpt.pweight_prelec(option, pars) for option in problem])
        values = np.array([cpt.value_fnc(option[:,0], pars) for option in problem])
        v = np.array([np.multiply(weights[i], values[i]) for i in range(len(problem))])
        evar = np.array([np.dot(weights[i], values[i] ** 2) - np.sum(v[i]) ** 2 for i in range(len(problem))])
        sigma2 = np.max([np.sum(evar), 1e-10])


        threshold = theta * np.sqrt(sigma2)
        thresholds[p] = threshold

        Z = np.zeros(N)

        # sequential weights
        omega = []
        for i, option in enumerate(problem):
            w = weights[i]
            omega.append(w * 1./option[:,1])
        omega = np.array(omega)

        for i in np.arange(N):

            prob_exp = deepcopy(problem)
            for opt in prob_exp:
                opt[:,1] = 0

            # sample initial starting point
            #pref = threshold + 1
            #while pref > threshold or pref < -threshold:
            #    pref = np.random.laplace(0, tau)
            pref = 0

            traces[p].append([deepcopy(pref)])

            n_steps = int(np.floor(np.random.normal(15, 3)))

            if np.random.random() < .5:
                first2 = [0, 1]
            else:
                first2 = [1, 0]

            steps = 0
            #while (pref < threshold and pref > -threshold) or steps < 2:
            while (steps < n_steps):

                steps += 1

                if steps==2 and (pref <= -threshold or pref >= threshold):
                    pass
                else:

                    # make sure each option is sampled at least once
                    if steps==1:
                        opt = first2[0]
                    elif steps==2:
                        opt = first2[1]
                    else:
                        # randomly pick an option to sample
                        opt = np.random.choice([0, 1])
                    #opt = np.random.choice([0, 1])
                    sgn = [-1, 1][opt]

                    # generate and evaluate an outcome
                    outcome_i = np.random.choice(range(problem[opt].shape[0]), 1,
                                                    p=problem[opt,:,1])[0]
                    outcome = problem[opt,outcome_i,0]

                    # increment the relative frequency for this outcome
                    prob_exp[opt,outcome_i,1] += 1

                    if np.random.random() > p_stay:
                        # weighting
                        u = cpt.value_fnc(outcome, pars)
                        om = omega[opt,outcome_i]

                        # update preference
                        #pref += (sgn*om*(u-phi))
                        pref += (sgn*(u-phi))

                    traces[p][-1].append(deepcopy(pref))

            #if pref <= -threshold:
            #    choice = 0
            #elif pref >= threshold:
            #    choice = 1
            #else:
            #    print pref, threshold

            if pref > 0:
                choice = 1
            elif pref < 0:
                choice = 0
            else:
                choice = np.random.choice([0, 1])

            samplesize[p][i] = steps
            choices[p][i] = choice

            for opt in prob_exp:
                opt[:,1] = opt[:,1]/opt[:,1].sum()
            problems_exp[p].append(prob_exp)

    if trace:
        return samplesize, choices, traces, thresholds
    elif relfreq:
        return samplesize, choices, problems_exp
    else:
        return samplesize, choices


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

