import os
from utils import checkpath, sim_id_str, bic
import pandas as pd
import numpy as np
from itertools import product
from copy import deepcopy
from random import uniform
from scipy.optimize import minimize
from collections import OrderedDict


def freepars(parset, PARS):
    fitting = {}
    for p in parset:
        if p.count('(') > 0:
            fitting[p] = PARS[p.split('(')[0]]
        else:
            fitting[p] = PARS[p]
    return fitting


def fit_mlh(model, problems, data, name,
            fixed={}, fitting={}, niter=5,
            outdir='.', method='Nelder-Mead', save=True, quiet=False):
    """Use maximum likelihood to fit CHASE model"""
    sim_id = sim_id_str(name, fixed, fitting)
    checkpath(outdir)

    cols = ['iteration', 'success', 'nllh', 'k', 'N', 'bic']

    # get range of thetas for grid search
    thetas = filter(lambda k: k.count('theta') > 0, fitting.keys())
    if len(thetas) > 0:
        theta_min, theta_max = fitting['theta']
        theta_prod = map(list, list(product(range(theta_min, theta_max + 1), repeat=len(thetas))))
        cols += thetas
    else:
        theta_prod = [[fixed['theta']]]
        cols += ['theta']

    rest = filter(lambda p: p.count('theta')==0, fitting.keys())
    rest.sort()
    cols += rest

    # determine number of parameters and observations
    k = len(fitting)
    N = data.shape[0]

    # create fit table
    arr = []
    for i in range(niter):
        for th in theta_prod:
            arr.append([i, np.nan, np.nan, k, N, np.nan] + th + [np.nan for _ in range(k - len(thetas))])
    fitdf = pd.DataFrame(arr, columns=cols)

    # iterate through parameter combinations
    for i, row in fitdf.iterrows():

        # update pars with current values of theta
        pars = deepcopy(fixed)
        for th in thetas:
            pars[th] = row[th]

        pars['fitting'] = OrderedDict([(p, fitting[p]) for p in rest])

        # if theta=1, can't fit tau
        if len(thetas)==1 and row[th]==1 and 'tau' in pars['fitting'] and 'stepsize' not in fixed:
            del pars['fitting']['tau']


        init = []
        for p in pars['fitting']:
            # if fitting normal stopping distribution, initialize at mean
            if p=='mu':
                init.append(data.samplesize.mean())
            else:
                init.append(uniform(fitting[p][0], fitting[p][1]))

        # fit!
        f = minimize(model.nloglik_opt, init, (problems, data, pars,),
                     method=method, options={'ftol': .001})

        fitdf.ix[i,'success'] = f['success']
        fitdf.ix[i,'nllh'] = f['fun']
        fitdf.ix[i,'bic'] = bic(f['fun'], k, N)
        for v, p in enumerate(pars['fitting'].keys()):
            fitdf.ix[i,p] = f['x'][v]

        if not quiet:
            print sim_id
            print '%s/%s' % (i, fitdf.shape[0])
            print '%s: %s' % (thetas, row[thetas].values)
            print fitdf.ix[i]

    # save the table
    if save: fitdf.to_csv('%s/%s.csv' % (outdir, sim_id))
    return fitdf


def load_results(name, fixed={}, fitting={}, outdir='.'):
    sim_id = sim_id_str(name, fixed, fitting)
    pth = '%s/%s.csv' % (outdir, sim_id)
    if os.path.exists(pth):
        return pd.read_csv(pth)
    else:
        print 'no file found: %s' % pth
        return []


def best_result(name, fixed={}, fitting={}, outdir='.', nopars=False):
    sim_id = sim_id_str(name, fixed, fitting)
    fitdf = load_results(name, fixed=fixed, fitting=fitting, outdir=outdir)
    fitdf = fitdf[fitdf.success==True].sort_values(by='nllh').reset_index()
    fitdf['sim_id'] = sim_id
    if fitdf.shape[0] == 0:
        return pd.Series({'sim_id': sim_id})
    if nopars:
        r = fitdf.ix[0,['sim_id', 'nllh', 'k', 'N', 'bic']]
        return r
    else:
        return fitdf.ix[0]


def predict_from_result(model, problems, data, name, fixed={}, fitting={},
                        groups=None, outdir='.', max_T=100):
    """Get predictions for fitted CHASE model"""

    # minimum sample size
    minsamplesize = fixed.get('minsamplesize', 1)

    # load the best result
    best = best_result(name, fixed, fitting, outdir=outdir)

    # free parameters that are not specific to any groups
    nonspec = filter(lambda k: k.count('(')==0, fitting.keys())
    for k in nonspec:
        data.loc[:,k] = [best[k] for _ in range(data.shape[0])]

    # free parameters that differ across groups
    factors = []
    spec = filter(lambda k: k.count('(')>0, fitting.keys())
    for k in spec:
        sp = k.rstrip(')').split('(')
        p = sp[0]
        f, value = sp[1].split('=')
        factors.append(f)
        ss = data[data[f]==value]
        data.loc[data[f]==value,p] = [best[k] for _ in range(ss.shape[0])]

    # get model predictions for each row in dataset
    arr = []
    for i, row in data.iterrows():

        pars = deepcopy(fixed)
        pars['max_T'] = max_T
        pid = row['problem']
        pars['probid'] = pid

        for k in nonspec:
            pars[k] = best[k]
        for k in spec:
            p = k.split('(')[0]
            pars[p] = row[p]

        # run the model
        r = model(problems[pid], pars)
        arr.append([pid, np.round(r['p_resp'][1], 3)] + \
                   list(minsamplesize + pred_quantiles(r)) + \
                   list(minsamplesize + pred_quantiles_by_choice(0, r)) + \
                   list(minsamplesize + pred_quantiles_by_choice(1, r)))

    preddf = pd.DataFrame(arr, columns=['problem', 'cp', 'ss(.25)', 'ss(.5)', 'ss(.75)',
                                        'ss_L(.25)', 'ss_L(.5)', 'ss_L(.75)',
                                        'ss_H(.25)', 'ss_H(.5)', 'ss_H(.75)'])
    return preddf


def pred_quantiles(pred, quantiles=[.25, .5, .75]):
    dist = np.sum([pred['p_resp'][i]*pred['p_stop_cond'][:,i] for i in [0,1]], axis=0)
    return np.array([np.sum(np.cumsum(dist) <= q) for q in quantiles])


def pred_quantiles_by_choice(option_ind, pred, quantiles=[.25, .5, .75]):
    dist = pred['p_stop_cond'][:,option_ind]
    q = np.array([np.sum(np.cumsum(dist) <= q) for q in quantiles])
    return q


#def pred_quantiles_all(pred, quantiles=[.25, .5, .75]):
#    arr = []
#    for probid in pred.keys():
#        dist = np.sum([pred[probid]['p_resp'][i]*pred[probid]['p_stop_cond'][:,i] for i in [0,1]], axis=0)
#        arr.append(np.array([np.sum(np.cumsum(dist) <= q) for q in quantiles]))
#    return np.mean(arr, axis=0)
