import numpy as np
import pandas as pd
from copy import deepcopy
import scipy.integrate as integrate
from utils import *
from collections import OrderedDict
from scipy.optimize import minimize
from random import uniform
from time import time


cached_options = {}
cached_values = {}


def value_fnc(x, pars):
    """Value weighting function

    x: array of outcomes
    pow_gain: exponent for gains (default=1)
    pow_loss: exponent for losses (default=pow_gain)
    w_loss: loss scaling parameter (default=1)
    """
    pow_gain = pars.get('pow_gain', 1.)
    pow_loss = pars.get('pow_loss', pow_gain)
    w_loss   = pars.get('w_loss', 1.)
    gain     = (x * (x >= 0.)) ** pow_gain
    loss     = -w_loss * ((-1 * (x * (x < 0.))) ** pow_loss)
    return gain + loss


def w_prelec(p, delta, gamma):
    """Prelec decision weighting function

    p:     array of probabilities
    delta: elevation parameter
    gamma: curvature parameter
    """
    assert np.all(p >= 0) and np.all(p <= 1)
    f = np.exp(-delta * ((-np.log(p)) ** gamma))

    # truncate at [0, 1]
    return np.clip(f, 0, 1)


def w_prime(p, delta, gamma):
    """Derivative of Prelec weighting function

    p:      array of probabilities
    delta:  elevation parameter
    gamma:  curvature parameter
    """
    return (delta*gamma/p) * ((-np.log(p)) ** (gamma - 1)) * w_prelec(p, delta, gamma)


def rank_outcomes_by_domain(option):
    """Create two pandas dataframes for
    ranked outcomes, separated by gains/losses"""
    gaindf = None
    lossdf = None
    gains  = []
    losses = []

    # separate gains and losses
    for i, opt in enumerate(option):
        if opt[0] > 0:
            gains.append([i, opt[0], opt[1], np.nan])
        elif opt[0] < 0:
            losses.append([i, opt[0], opt[1], np.nan])

    # if there are x=0 outcomes but no gains, group with losses
    for i, opt in enumerate(option):
        if opt[0] == 0:
            if len(gains) == 0:
                losses.append([i, opt[0], opt[1], np.nan])
            else:
                gains.append([i, opt[0], opt[1], np.nan])

    if len(gains) > 0:
        gaindf = pd.DataFrame(np.array(gains), columns=['id', 'outcome', 'pr', 'w']).sort_values(by='outcome').reset_index()
    if len(losses) > 0:
        lossdf = pd.DataFrame(np.array(losses), columns=['id', 'outcome', 'pr', 'w']).sort_values(by='outcome').reset_index()

    return gaindf, lossdf


def pweight_prelec(option, pars):
    prelec_elevation = pars.get('prelec_elevation', 1.)
    prelec_gamma = pars.get('prelec_gamma', 1.)
    prelec_elevation_loss = pars.get('prelec_elevation_loss', prelec_elevation)
    prelec_gamma_loss = pars.get('prelec_gamma_loss', prelec_gamma)

    if 'gaindf' in option:
        gaindf = deepcopy(option['gaindf'])
        lossdf = deepcopy(option['lossdf'])
    else:
        gaindf, lossdf = rank_outcomes_by_domain(option)
    n_gains = gaindf.shape[0] if gaindf is not None else 0
    n_losses = lossdf.shape[0] if lossdf is not None else 0

    if n_gains > 0:
        q = gaindf.pr.values
        r = np.append(np.cumsum(q[::-1])[::-1], [0])
        wr = w_prelec(r, prelec_elevation, prelec_gamma)
        wrd = -np.ediff1d(wr)
        gaindf.w = wrd

    if n_losses > 0:
        q = lossdf.pr.values
        r = np.append([0], np.cumsum(q))
        wr = w_prelec(r, prelec_elevation_loss, prelec_gamma_loss)
        wrd = np.ediff1d(wr)
        lossdf.w = wrd

    print q
    print r
    print wr
    print wrd



    # put ranked weights back in original order
    weights = np.zeros(n_gains + n_losses)
    for i in range(n_gains):
        weights[int(gaindf.iloc[i]['id'])] = gaindf.iloc[i]['w']
    for i in range(n_losses):
        weights[int(lossdf.iloc[i]['id'])] = lossdf.iloc[i]['w']

    try:
        assert not np.any(np.isnan(weights)) and np.sum(weights) > 0
    except AssertionError:
        print prelec_gamma, prelec_elevation
        print option
        print weights
        print gaindf
        print lossdf
        print dummy

    # normalize (in case of mixed option)
    weights = weights / np.sum(weights)
    weights = np.clip(weights, 0, 1)
    return weights


def setup(problems):
    rdw = {}
    for pid in problems:
        options = problems[pid]
        rdw[pid] = []
        for option in options:
            gaindf, lossdf = rank_outcomes_by_domain(option)
            rdw[pid].append({'gaindf': gaindf, 'lossdf': lossdf})
    return rdw


def normal_raised_to_power(option, alpha):
    """For an option with normally distributed outcomes X,
    find the expected value and expected variance after
    transformation X^alpha"""
    mu, sigma2 = option

    def pdf(x):
        return (1/np.sqrt(2*np.pi*(sigma2)))*np.exp(-((x - mu)**2)/(2*sigma2))

    def f_positive(x):
        return x**alpha * pdf(x)

    def f_negative(x):
        return -((x - 2*mu)**(alpha)) * pdf(x)

    def f_var_positive(x):
        return x**(2*alpha) * pdf(x)

    def f_var_negative(x):
        return (((x - 2*mu)**(2*alpha))) * pdf(x)

    # integrate f_positive over [0, np.inf], f_negative over [2*mu, np.inf]
    sPos = integrate.quad(lambda x: f_positive(x), 0, np.inf)[0]
    sNeg = integrate.quad(lambda x: f_negative(x), mu*2, np.inf)[0]

    # integrate f_var_positive over [0, np.inf], f_var_negative over [2*mu, np.inf]
    vPos = integrate.quad(lambda x: f_var_positive(x), 0, np.inf)[0]
    vNeg = integrate.quad(lambda x: f_var_negative(x), mu*2, np.inf)[0]

    ev = sPos + sNeg
    evar = vPos + vNeg - ev**2
    return ev, evar


def choice_prob(options, pars, problemid=None, use_cache=False):
    global cached_options, cached_values

    options_str = np.array_str(options)
    if options_str in cached_values and use_cache:
        cp = cached_values[options_str]
    else:

        s = pars.get('s', 1.) # choice sensitivity

        if 'prelec_gamma' in pars or 'prelec_elevation' in pars:
            if problemid != None: wopt = cached_options[problemid]
            else:                 wopt = options
            weights = np.array([pweight_prelec(option, pars) for option in wopt])
        else:
            weights = options[:,:,1]

        if 'pow_gain' in pars or 'w_loss' in pars:
            values = np.array([value_fnc(option[:,0], pars) for option in options])
        else:
            values = options[:,:,0]

        vL, vH = [np.dot(w, v) for (w, v) in zip(weights, values)]
        cp = 1/(1. + np.exp(-s * (vH - vL)))
        assert not np.isnan(cp)

        cached_values[options_str] = cp

    return cp


def MSD(value, problems, data, args):
    global cached_values

    # reset cache for new value
    cached_values = {}

    pars, fitting, verbose = unpack(value, args)
    if outside_bounds(value, fitting): return np.inf
    observed = data.groupby('problem').apply(lambda d: d.choice.mean()).values
    predicted = np.array([choice_prob(problems[k], pars, problemid=k) for k in problems])
    msd = np.mean((predicted - observed) ** 2)
    return msd


def fit_msd(problems, data, name, fixed={}, fitting={}, niter=1, outdir='.'):
    """Use maximum likelihood to fit CPT choice model"""
    global cached_options
    # setup dataframes in advance
    cached_options = setup(problems)
    sim_id = sim_id_str(name, fixed, fitting)
    print sim_id
    checkpath(outdir)

    cols = ['iteration', 'success', 'k', 'N', 'msd']
    rest = fitting.keys()
    rest.sort()
    cols += rest

    # determine number of parameters and observations
    k = len(fitting)
    N = data.shape[0]

    # create fit table
    arr = []
    for i in range(niter):
        arr.append([i, np.nan, k, N, np.nan] + [np.nan for _ in range(k)])
    fitdf = pd.DataFrame(arr, columns=cols)


    # iterate through
    for i, row in fitdf.iterrows():

        print '%s/%s' % (i, fitdf.shape[0])

        # update pars with current values of theta
        pars = deepcopy(fixed)
        pars['fitting'] = OrderedDict([(p, fitting[p]) for p in rest])
        init = []
        for p in rest:
            if len(fitting[p]) == 3:
                init.append(fitting[p][2])
            else:
                init.append(uniform(fitting[p][0], fitting[p][1]))

        f = minimize(MSD, init, (problems, data, pars,),
                     method='Nelder-Mead', options={'ftol': 1e-8, 'xtol': 1e-8})

        fitdf.ix[i,'success'] = f['success']
        fitdf.ix[i,'msd'] = f['fun']
        for v, p in enumerate(pars['fitting'].keys()):
            fitdf.ix[i,p] = f['x'][v]

    # save the table
    fitdf.to_csv('%s/%s.csv' % (outdir, sim_id))
    return fitdf


def nloglik_across_gambles(value, problems, data, args):
    pars, fitting, verbose = unpack(value, args)
    if outside_bounds(value, fitting):
        return np.inf
    else:

        pids = data.problem.unique()

        predicted = {pid: choice_prob(problems[pid], pars) for pid in problems if pid in pids}
        cp = np.array([predicted[pid] for pid in data.problem.values])
        choice = data.choice.values
        llh = np.sum(np.log(pfix(cp[choice==1]))) + \
              np.sum(np.log(pfix(1-cp[choice==0])))

        #print value, -llh
        return -llh


def fit_llh(data, fixed={}, fitting=[]):

    pars = {'data': data}
    for p in fixed:
        pars[p] = fixed[p]
    pars['fitting'] = fitting

    init = [randstart(par) for par in pars['fitting']]
    f = minimize(nloglik_across_gambles, init, (pars,), method='Nelder-Mead')

    return {'bf_par': {fitting[i]: f['x'][i] for i in range(len(fitting))},
            'nllh': f['fun'],
            'bic': bic(f, pars),
            'success': f['success']}


def fit(problems, data, name, fixed={}, fitting={}, niter=1, outdir='.', opt='llh'):
    """Use maximum likelihood to fit CPT model"""
    global cached_options
    # setup dataframes in advance
    cached_options = setup(problems)
    sim_id = sim_id_str(name, fixed, fitting)
    print sim_id
    checkpath(outdir)

    cols = ['iteration', 'success', 'k', 'N', 'msd', 'llh']
    rest = fitting.keys()
    rest.sort()
    cols += rest

    # determine number of parameters and observations
    k = len(fitting)
    N = data.shape[0]

    # create fit table
    arr = []
    for i in range(niter):
        arr.append([i, np.nan, k, N, np.nan, np.nan] + [np.nan for _ in range(k)])
    fitdf = pd.DataFrame(arr, columns=cols)

    if opt is 'llh':
        func = nloglik_across_gambles
    elif opt is 'msd':
        func = MSD

    # iterate through
    for i, row in fitdf.iterrows():

        #print '%s/%s' % (i, fitdf.shape[0])

        # update pars with current values of theta
        pars = deepcopy(fixed)
        pars['fitting'] = OrderedDict([(p, fitting[p]) for p in rest])
        init = []
        for p in rest:
            if len(fitting[p]) == 3:
                init.append(fitting[p][2])
            else:
                init.append(uniform(fitting[p][0], fitting[p][1]))

        f = minimize(func, init, (problems, data, pars,),
                     method='Nelder-Mead', options={'ftol': .0001})

        fitdf.ix[i,'success'] = f['success']
        fitdf.ix[i,opt] = f['fun']
        for v, p in enumerate(pars['fitting'].keys()):
            fitdf.ix[i,p] = f['x'][v]

    # save the table
    fitdf.to_csv('%s/%s.csv' % (outdir, sim_id))
    return fitdf
