import numpy as np
import cpt
from copy import deepcopy
from time import time
from utils import *
from chase.base import *
from scipy.stats import truncnorm, geom, laplace, norm

X_MIN = -100
X_MAX = 180


class CHASEProcessModel(object):


    def __init__(self, **kwargs):
        self.problems = kwargs.get('problems', None)
        self.problemtype = kwargs.get('problemtype', 'multinomial')
        self.stoprule = kwargs.get('stoprule', 'optional')
        self.choicerule = kwargs.get('choicerule', None)

        # setup options for weighting functions
        if self.problemtype is 'multinomial' and self.problems is not None:
            self.rdw = cpt.setup(self.problems)


    def __call__(self, options, pars, obs=None, trackobs=False):
        """Simulate process model to get predicted
        choice and sample size distributions"""


        ### Basic setup
        np.random.seed()
        N     = pars.get('N', 10000)   # number of simulated trials
        max_T = pars.get('max_T', 1000) # maximum sample size


        ### Stopping rules

        if self.stoprule == 'optional':
            threshold  = pars.get('theta', 3)   # decision threshold (optional only)
            r      = pars.get('r', 0)       # rate of boundary collapse (optional only)
            stop_T = None

        # fixed sample size
        elif self.stoprule=='fixedT':
            stop_T = pars.get('stop_T', 2)
            max_T  = stop_T
            threshold = 1000

        # geometric
        elif self.stoprule=='fixedGeom':

            threshold = 1000
            p_stop_geom = pars.get('p_stop_geom')
            minss = pars.get('minsamplesize', 1)

            # sample size (not index), adjusted by minsamplesize
            stop_T = geom.rvs(p_stop_geom, size=N) + (minss - 1)

            # don't go past max_T
            stop_T[np.where(stop_T > max_T)[0]] = max_T


        ### Search

        # probability of sampling each option
        p_sample_H = pars.get('p_sample_H', .5)
        p_sample_L = 1 - p_sample_H

        # if p_switch is specified, it will be used to generate
        # sequences of observations (rather than p_sample_H and p_sample_L)
        p_switch   = pars.get('p_switch', None)

        # are the first two samples drawn from different options?
        switchfirst = pars.get('switchfirst', False)


        ### Sequential weights

        # compute value and attentional weights for multinomial problems
        if self.problemtype is 'multinomial':
            if self.rdw is None: wopt = options
            else:                wopt = self.rdw[pars['probid']]
            weights = np.array([cpt.pweight_prelec(option, pars) for option in wopt])
            values = np.array([cpt.value_fnc(option[:,0], pars) for option in options])
            v = np.array([np.multiply(weights[i], values[i]) for i in range(len(options))])
            V = v.sum(axis=1)
            evar = np.array([np.dot(weights[i], values[i] ** 2) - np.sum(v[i]) ** 2 for i in range(len(options))])
            sigma2 = np.max([np.sum(evar), 1e-10])
            sigma2mean = np.max([np.mean(evar), 1e-10])

            # sequential weights
            omega = []
            for i, option in enumerate(options):
                omega.append(weights[i]/option[:,1])
            omega = np.array(omega)
            omega[np.isnan(omega)] = 0
            w_outcomes = np.array([np.multiply(omega[i], values[i]) for i in range(len(options))])

        elif self.problemtype is 'normal':

            if 'pow_gain' in pars:
                w_options = np.array([[0,0],[0,0]])
                for i in range(2):
                    ev, evar = cpt.normal_raised_to_power(options[i], pars['pow_gain'])
                    w_options[i] = np.array([ev, evar])
                sigma2 = w_options[:,1].sum()
                evar = w_options[:,1]
            else:
                evar = options[:,1]
                sigma2 = options[:,1].sum()
                sigma2mean = options[:,1].mean()

        # scale by variance
        if 'sc' in pars:
            # raised to power
            sc = pars.get('sc')
            variance_scale = 1 / float(np.sqrt(sigma2) ** sc)
        elif 'sc2' in pars:
            # multiplicative
            sc = pars.get('sc2')
            variance_scale = 1 / float(np.sqrt(sigma2) * sc)
        elif 'sc0' in pars:
            sc0 = pars.get('sc0')
        elif 'sc_mean' in pars:
            sc = pars.get('sc_mean')
            variance_scale = 1 / float(np.sqrt(sigma2mean) ** sc)
        elif 'sc2_mean' in pars:
            sc = pars.get('sc2_mean')
            variance_scale = 1 / float(np.sqrt(sigma2mean) * sc)
        elif 'sc_x' in pars:
            variance_scale = pars.get('sc_x')
        else:
            variance_scale = 1


        ### Starting distribution

        Z = np.zeros(N)
        if 'tau' in pars:
            tau = pars.get('tau')
            Z = laplace.rvs(loc=0, scale=tau, size=N)

        elif 'tau_trunc' in pars:
            tau = pars.get('tau_trunc')
            dx = .001
            x = np.arange(-(threshold-dx), threshold, dx)
            p = laplace.pdf(x, loc=0, scale=tau)
            pn = p/p.sum()
            Z = np.random.choice(x, N, p=pn)

        elif 'tau_rel' in pars:
            tau = pars.get('tau_rel')
            tau = tau / variance_scale
            Z = laplace.rvs(loc=0, scale=tau, size=N)

        elif 'tau_rel_trunc' in pars:
            tau = pars.get('tau_rel_trunc')
            dx = .001
            x = np.arange(-1+dx, 1, dx)
            p = laplace.pdf(x, loc=0, scale=tau)
            pn = p/p.sum()
            Z = np.random.choice(x, N, p=pn)
            Z = Z * threshold

        elif 'tau_unif' in pars:
            #tau = pars.get('tau_unif', .001)
            #theta_max = pars.get('theta_max', theta)
            #theta_max = 200
            #rng = tau * theta_max
            rng = pars.get('tau_unif', .001)
            Z = np.linspace(-rng, rng, num=N)
            np.random.shuffle(Z)
            #Z = np.random.uniform(low=(-tau), high=tau, size=N)

        elif 'tau_unif_rel' in pars:
            dx = .001
            rng = pars.get('tau_unif_rel', .001)
            Z = np.linspace(-(threshold-dx) * rng, (threshold-dx) * rng, num=N)
            np.random.shuffle(Z)

        elif 'tau_normal' in pars:
            tau = pars.get('tau_normal')
            Z = norm.rvs(loc=0, scale=tau, size=N)

        elif 'tau_normal_trunc' in pars:
            tau = pars.get('tau_normal_trunc')
            dx = .001
            x = np.arange(-(threshold-dx), threshold, dx)
            p = norm.pdf(x, loc=0, scale=tau)
            pn = p/p.sum()
            Z = np.random.choice(x, N, p=pn)


        ### Simulate

        if obs is not None:

            # assume a single sequence of known observations
            sampled_option = obs['option'].values
            outcomes       = obs['outcome'].values
            max_T          = outcomes.shape[0]
            sgn            = 2*sampled_option - 1
            sv             = np.zeros(outcomes.shape)

            if self.problemtype is 'normal':

                c = pars.get('c', 0)

                # add weighting and criterion here
                sv = cpt.value_fnc(outcomes - c, pars)

            elif self.problemtype is 'multinomial':
                pass
                for i, opt in enumerate(options):
                    for j, x in enumerate(opt):
                        ind = np.where((sampled_option==i) & (outcomes==x[0]))[0]
                        sv[ind] = w_outcomes[i][j]

            sv = np.multiply(sv, sgn)
            sampled_option = np.tile(sampled_option, (N, 1))
            outcomes = np.tile(outcomes, (N, 1))
            sv = np.tile(sv, (N, 1))


        elif self.choicerule == 'random':
            sv = np.zeros((N, max_T))
            sampled_option = None
            outcomes = None

        else:
            # otherwise, simulate sampling from options

            if False and not trackobs and self.problemtype is 'multinomial' and p_switch is None:
                sampled_option = None
                outcomes = None

                valence = deepcopy(w_outcomes)
                valence[0] = -1 * valence[0]
                valence = valence.ravel()

                p = deepcopy(options[:,:,1])
                p[0] = p_sample_L * p[0]
                p[1] = p_sample_H * p[1]
                p = p.ravel()

                sv = np.random.choice(valence, p=p, size=(N, max_T))

                # ensure that both options are sampled
                # at least once
                if switchfirst:
                    first = np.random.binomial(1, .5, size=N)
                    second = 1 - first
                    first2 = np.transpose((first, second))
                    sampled_A = first2==0
                    sampled_B = first2==1

                    observed_A = np.random.choice(range(len(w_outcomes[0])),
                                                size=sampled_A.sum(),
                                                p=options[0][:,1])
                    observed_B = np.random.choice(range(len(w_outcomes[1])),
                                                size=sampled_B.sum(),
                                                p=options[1][:,1])

                    # subjective weighting
                    sv2 = np.zeros((N, 2))
                    sv2[sampled_A] = -1 * w_outcomes[0][observed_A]
                    sv2[sampled_B] =      w_outcomes[1][observed_B]
                    sv[:,:2] = sv2

            else:

                # which option was sampled
                sampled_option = np.zeros((N, max_T), int)

                if p_switch is None:
                    # ignore switching, just search based on [p_sample_H, p_sample_L]
                    sampled_option = np.random.binomial(1, p_sample_H, size=(N, max_T))
                else:
                    # generate search sequences based on p_switch
                    switches = np.random.binomial(1, p_switch, size=(N, max_T - 1))
                    sampled_option[:,0] = np.random.binomial(1, .5, size=N)
                    for i in range(max_T - 1):
                        switch_i = switches[:,i]
                        sampled_option[:,i+1] = np.abs(sampled_option[:,i] - switch_i)

                # ensure both options sampled at least once
                if switchfirst:
                    first = np.random.binomial(1, .5, size=N)
                    sampled_option[:,0] = first
                    sampled_option[:,1] = 1 - first

                # FOR SIMULATION
                #sampled_option = np.zeros((N, max_T), int)
                #for i in range(N):
                #    arr = sampled_option[i]
                #    arr[:(max_T/2)] = 1
                #    np.random.shuffle(arr)
                #    sampled_option[i] = arr


                # FOR SIMULATION
                #p_switch = pars.get('p_switch', .5)

                #sampled_option = np.zeros((N, max_T), int)
                #sampled_option[:,0] = np.random.choice([0, 1], p=[.5, .5], size=N)

                #for i in range(max_T - 1):
                #    switch = np.random.choice([0, 1], p=[1-p_switch, p_switch], size=N)
                #    sampled_option[:,i+1] = np.abs(sampled_option[:,i] - switch)


                sampled_A = sampled_option==0
                sampled_B = sampled_option==1
                N_sampled_A = sampled_A.sum()
                N_sampled_B = sampled_B.sum()

                # observation matrix - which outcome occurred (by index)
                observed = np.zeros((N, max_T), int)
                if self.problemtype is 'multinomial':
                    observed_A = np.random.choice(range(len(w_outcomes[0])),
                                                size=sampled_A.sum(),
                                                p=options[0][:,1])
                    observed_B = np.random.choice(range(len(w_outcomes[1])),
                                                size=sampled_B.sum(),
                                                p=options[1][:,1])
                    observed[sampled_A] = observed_A
                    observed[sampled_B] = observed_B


                # record outcomes experienced (by value)
                outcomes = np.zeros((N, max_T))
                if self.problemtype is 'multinomial':
                    obj_outcomes = options[:,:,0]
                    #outcomes[sampled_A] = obj_outcomes[0][observed_A]
                    #outcomes[sampled_B] = obj_outcomes[1][observed_B]

                    # note weighting already done above
                    outcomes[sampled_A] = w_outcomes[0][observed_A]
                    outcomes[sampled_B] = w_outcomes[1][observed_B]
                    outcomes_A = outcomes[sampled_A]
                    outcomes_B = outcomes[sampled_B]
                else:
                    A, B = options
                    sigmaA = np.sqrt(A[1])
                    sigmaB = np.sqrt(B[1])

                    # weird conversion for np.truncnorm
                    lowerA, upperA = (X_MIN - A[0]) / sigmaA, (X_MAX - A[0]) / sigmaA
                    lowerB, upperB = (X_MIN - B[0]) / sigmaB, (X_MAX - B[0]) / sigmaB
                    outcomes_A = np.round(truncnorm.rvs(lowerA, upperA, loc=A[0], scale=sigmaA, size=N_sampled_A))
                    outcomes_B = np.round(truncnorm.rvs(lowerB, upperB, loc=B[0], scale=sigmaB, size=N_sampled_B))
                    outcomes[sampled_A] = outcomes_A
                    outcomes[sampled_B] = outcomes_B

                    if 'pow_gain' in pars:
                        outcomes   = cpt.value_fnc(outcomes, pars)
                        outcomes_A = cpt.value_fnc(outcomes_A, pars)
                        outcomes_B = cpt.value_fnc(outcomes_B, pars)


                # comparison
                sv = np.zeros((N, max_T))


                # criteria for each option
                if 'c' in pars:
                    # compare to constant
                    c = pars.get('c')
                    c_A = c * np.ones(outcomes_A.shape)
                    c_B = c * np.ones(outcomes_B.shape)

                elif 'c_0' in pars:
                    # compare to sample mean
                    c_0 = pars.get('c_0', 45)

                    sum_A = np.cumsum(np.multiply(sampled_A, outcomes), axis=1)
                    N_A = np.cumsum(sampled_A, axis=1, dtype=float)
                    mn_A = np.multiply(sum_A, 1/N_A)
                    mn_A[np.isnan(mn_A)] = c_0

                    sum_B = np.cumsum(np.multiply(sampled_B, outcomes), axis=1)
                    N_B = np.cumsum(sampled_B, axis=1, dtype=float)
                    mn_B = np.multiply(sum_B, 1/N_B)
                    mn_B[np.isnan(mn_B)] = c_0

                    compA = np.multiply(outcomes - mn_B, sampled_A)
                    compB = np.multiply(outcomes - mn_A, sampled_B)
                    #sv = (-1 * compA) + compB

                else:
                    # (default) compare to true (weighted)
                    # mean of other option
                    if self.problemtype is 'multinomial':
                        A, B = V
                    elif self.problemtype is 'normal':
                        if 'pow_gain' in pars:
                            A, B = w_options[:,0]
                        else:
                            A, B = options[:,0]
                    c_A = B * np.ones(outcomes_A.shape)
                    c_B = A * np.ones(outcomes_B.shape)

                # combine
                if 'c_0' in pars:
                    sv = (-1 * compA) + compB
                else:
                    sv[sampled_A] = -1 * (outcomes_A - c_A)
                    sv[sampled_B] =      (outcomes_B - c_B)


                if 'sc0' in pars:

                    # for any options with a variance of zero,
                    # replace with sc0
                    evar[evar==0.] = sc0

                    # scaling factor for each option depends on
                    # its variance
                    sc_A, sc_B = 1/np.sqrt(evar)
                    sv[sampled_A] = sv[sampled_A] * sc_A
                    sv[sampled_B] = sv[sampled_B] * sc_B
                else:
                    # fixed scaling factor across all options
                    sv = sv * variance_scale


                # noise
                if 'c_sigma' in pars:
                    c_sigma = pars.get('c_sigma')
                    err = np.random.normal(loc=0, scale=c_sigma, size=outcomes.shape)
                elif 'dv_sigma' in pars:
                    dv_sigma = pars.get('dv_sigma')
                    err = np.random.normal(loc=0, scale=dv_sigma, size=N)
                    err = np.tile(err, (max_T, 1)).transpose()
                else:
                    err = np.zeros(outcomes.shape)

                sv = sv + err

        ### Accumulation

        # add starting states to first outcome
        sv[:,0] = sv[:,0] + Z

        # p_stay
        #p_stay = pars.get('p_stay', 0)
        #if p_stay > 0:
        #    attended = np.random.binomial(1, 1-p_stay, size=(N, max_T))
        #    sv = np.multiply(sv, attended)


        # accumulate
        P = np.cumsum(sv, axis=1)


        ### Stopping

        if self.stoprule == 'optional':
            if r > 0:
                # collapsing boundaries
                threshold_min = .1
                upper = threshold_min * np.ones((N, max_T))
                dec = np.arange(threshold, threshold_min, -r*threshold)
                dec = dec[:max_T]
                upper[:,:dec.shape[0]] = np.tile(dec, (N, 1))

                lower = -threshold_min * np.ones((N, max_T))
                inc = np.arange(-threshold, -threshold_min, r*threshold)
                inc = inc[:max_T]
                lower[:,:inc.shape[0]] = np.tile(inc, (N, 1))

                crossed = -1 * (P < lower) + 1 * (P > upper)
            else:
                # fixed boundaries
                crossed = -1 * (P < -threshold) + 1 * (P > threshold)

            # if minimum sample size, prevent stopping
            minsamplesize = pars.get('minsamplesize', 1) - 1
            crossed[:,:minsamplesize] = 0

            # any trials where hit max_T, make decision based on
            # whether greater or less than zero
            nodecision = np.where(np.sum(np.abs(crossed), axis=1)==0)[0]
            if len(nodecision) > 0:
                n_pos = np.sum(P[nodecision,max_T-1] > 0)
                n_eq = np.sum(P[nodecision,max_T-1] == 0)
                n_neg = np.sum(P[nodecision,max_T-1] < 0)
                #assert n_eq == 0, "reached max_T with preference of 0"

                crossed[nodecision,max_T-1] +=  1*(P[nodecision,max_T-1] >= 0)
                crossed[nodecision,max_T-1] += -1*(P[nodecision,max_T-1] < 0)

        elif self.stoprule == 'fixedT':
            crossed = np.zeros((N, stop_T), dtype=int)
            crossed[:,(stop_T-1)] = np.sign(P[:,(stop_T-1)])

            indifferent = np.where(crossed[:,(stop_T-1)]==0)[0]
            n_indifferent = len(indifferent)
            crossed[indifferent] = np.random.choice([-1,1], p=[.5, .5], size=(n_indifferent,1))
            assert np.sum(crossed[:,(stop_T-1)]==0)==0

        elif self.stoprule == 'fixedGeom':

            crossed = np.zeros((N, max_T), dtype=int)
            crossed[range(N),stop_T-1] = np.sign(P[range(N),stop_T-1])

            indifferent = np.where(crossed[range(N),stop_T-1]==0)[0]
            n_indifferent = len(indifferent)
            t_indifferent = (stop_T-1)[indifferent]
            crossed[indifferent,t_indifferent] = np.random.choice([-1,1], p=[.5,.5], size=n_indifferent)


        if obs is not None:

            p_stop_choose_A = np.sum(crossed==-1, axis=0)*(1/float(N))
            p_stop_choose_B = np.sum(crossed==1, axis=0)*(1/float(N))
            p_sample = 1 - (p_stop_choose_A + p_stop_choose_B)

            return {'p_stop_choose_A': p_stop_choose_A,
                    'p_stop_choose_B': p_stop_choose_B,
                    'p_sample': p_sample,
                    'traces': P}

        else:

            # samplesize is the **index** where threshold is crossed
            samplesize = np.sum(1*(np.cumsum(np.abs(crossed), axis=1)==0), axis=1)
            choice = (crossed[range(N),samplesize] + 1)/2
            p_resp = choice.mean()
            ss_A = samplesize[choice==0]
            ss_B = samplesize[choice==1]

            p_stop_A = np.zeros(max_T)
            p_stop_B = np.zeros(max_T)
            if self.stoprule == 'optional' or self.stoprule == 'fixedGeom':
                p_stop_A_f = np.bincount(ss_A, minlength=max_T)
                if p_stop_A_f.sum() > 0:
                    p_stop_A = p_stop_A_f/float(p_stop_A_f.sum())
                p_stop_B_f = np.bincount(ss_B, minlength=max_T)
                if p_stop_B_f.sum() > 0:
                    p_stop_B = p_stop_B_f/float(p_stop_B_f.sum())

            elif self.stoprule == 'fixedT':
                p_stop_A[stop_T-1] = 1
                p_stop_B[stop_T-1] = 1

            assert (p_stop_A_f.sum() + p_stop_B_f.sum()) == N

            p_stop_cond = np.transpose([p_stop_A, p_stop_B])
            p_stop_cond[np.isnan(p_stop_cond)] = 0.
            f_stop_cond = np.transpose([p_stop_A_f, p_stop_B_f])/float(N)


            # only include data up to choice
            outcome_ind = None
            traces = None
            if type(sampled_option) is np.ndarray and trackobs:
                sampled_option = [sampled_option[i][:(samplesize[i]+1)] for i in range(samplesize.shape[0])]
                outcomes       = [outcomes[i][:(samplesize[i]+1)] for i in range(samplesize.shape[0])]
                traces         = [P[i][:(samplesize[i]+1)] for i in range(samplesize.shape[0])]
                if self.problemtype is 'multinomial':
                    outcome_ind    = [observed[i][:(samplesize[i]+1)] for i in range(samplesize.shape[0])]


            return {'choice': choice,
                    'samplesize': samplesize + 1,
                    'p_resp': np.array([1-p_resp, p_resp]),
                    'p_stop_cond': p_stop_cond,
                    'f_stop_cond': f_stop_cond,
                    'sampled_option': sampled_option,
                    'outcomes': outcomes,
                    'outcome_ind': outcome_ind,
                    'traces': traces,
                    'Z': Z
                    }


    def nloglik(self, problems, data, pars, choiceonly=False):
        """For a single set of parameters, evaluate the
        log-likelihood of observed data set."""


        nonspec = pars['nonspec']
        spec = pars['spec']
        factors = []
        for k in spec:
            sp = k.rstrip(')').split('(')
            p = sp[0]
            f, value = sp[1].split('=')
            if f not in factors: factors.append(f)
            data.loc[data[f]==value,p] = pars[k]

        nllh = []
        nllh_choice = []
        nllh_sample = []

        knownobs = pars.get('knownobs', False)
        if not knownobs:
            for i, probdata in data.groupby(['problem'] + factors):

                pid = probdata['problem'].values[0]
                grppars = {}
                for k in nonspec: grppars[k] = pars[k]
                grppars['max_T'] = probdata.samplesize.max() + 1
                grppars['probid'] = pid
                for k in spec:
                    p = k.split('(')[0]
                    grppars[p] = probdata[p].values[0]

                # run the model
                results = self.__call__(problems[pid], grppars)
                ss = np.array(probdata.samplesize.values, int) - 1
                choices = np.array(probdata.choice.values, int)

                # get the likelihood
                nllh_choice.append(sum_neg_log(results['f_stop_cond'][ss,choices]))

        else:
            for i, probdata in data.groupby(['subject', 'problem']):
                pid = probdata['problem'].values[0]
                pars['probid'] = pid
                grppars = {}
                for k in nonspec: grppars[k] = pars[k]

                spec = filter(lambda k: k.count('(')>0, pars)
                for k in spec:
                    p = k.split('(')[0]
                    grppars[p] = probdata[p].values[0]

                results = self.__call__(problems[pid], grppars, obs=probdata)

                choice = probdata.choice.values[0]
                p_choice = np.array([results['p_stop_choose_A'][-1], results['p_stop_choose_B'][-1]])[choice]
                p_sample = results['p_sample'][:-1]

                nllh_choice.append(-np.log(pfix(p_choice)))
                nllh_sample.append(-np.sum(np.log(pfix(p_sample))))

        if choiceonly:
            return np.sum(nllh_choice)
        else:
            return np.sum(nllh_choice) + np.sum(nllh_sample)


    def nloglik_opt(self, value, problems, data, pars):
        pars, fitting, verbose = unpack(value, pars)

        # check if the parameters are within allowed range
        if outside_bounds(value, fitting):
            return np.inf
        else:
            for v, p in zip(value, fitting.keys()):
                pars[p] = v

            start = time()
            nllh = self.nloglik(problems, data, pars)
            v = np.round(value, 2)
            t = np.round(time() - start, 2)
            #print '%s --> %s\t[time: %s]' % (v, np.round(nllh, 1), t)
            return nllh


if __name__=='__main__':


    problems = {}
    arr = np.genfromtxt('paper/data/six_problems.csv', delimiter=',')
    for i in range(len(arr)):
        problems[i] = arr[i].reshape((2,2,2))


    m = CHASEProcessModel(problems=problems)

    for pid in problems:
        pars = {'probid': pid,
                'p_stay': .3}
        m(problems[pid], pars)

