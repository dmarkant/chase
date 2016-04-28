import os
from utils import checkpath, sim_id_str, bic
import pandas as pd
import numpy as np
from itertools import product
from copy import deepcopy
from random import uniform
from scipy.optimize import minimize, fmin
from collections import OrderedDict


def fit_mlh(model, problems, data, name,
            fixed={}, fitting={}, niter=5,
            outdir='.', method='Nelder-Mead', save=True, quiet=False):
    """Use maximum likelihood to fit CHASE model"""
    sim_id = sim_id_str(name, fixed, fitting)
    checkpath(outdir)

    cols = ['iteration', 'success', 'nllh', 'k', 'N', 'bic']

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


    # iterate through
    for i, row in fitdf.iterrows():

        # update pars with current values of theta
        pars = deepcopy(fixed)
        for th in thetas:
            pars[th] = row[th]

        pars['fitting'] = OrderedDict([(p, fitting[p]) for p in rest])

        init = []
        for p in rest:
            if p=='mu':
                init.append(data.samplesize.mean())
            elif len(fitting[p]) == 3:
                init.append(fitting[p][2])
            else:
                init.append(uniform(fitting[p][0], fitting[p][1]))

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
        return pd.read_csv('%s/%s.csv' % (outdir, sim_id))
    else:
        return []


def best_result(name, fixed={}, fitting={}, outdir='.', nopars=False):
    sim_id = sim_id_str(name, fixed, fitting)
    fitdf = load_results(name, fixed=fixed, fitting=fitting, outdir=outdir)
    fitdf = fitdf[fitdf.success==True].sort('nllh').reset_index()
    fitdf['sim_id'] = sim_id
    if fitdf.shape[0] == 0:
        return pd.Series({'sim_id': sim_id})
    if nopars:
        r = fitdf.ix[0,['sim_id', 'nllh', 'k', 'N', 'bic']]
        return r
    else:
        return fitdf.ix[0]


def predict_from_result(model, problems, name, fixed={}, fitting={}, groups=None, outdir='.', max_T=300):

    # load the best result
    best = best_result(name, fixed, fitting, outdir=outdir)

    results = {}

    if groups==None:
        # copy best-fit parameter settings
        pars = deepcopy(fixed)
        pars['max_T'] = max_T

        for p in fitting:
            pars[p] = best[p]

        # run the model for each problem
        for pid in problems:
            pars['probid'] = pid
            results[pid] = model(problems[pid], pars)

        return results

    else:

        for grp in groups:

            # copy best-fit parameter settings
            pars = deepcopy(fixed)
            pars['max_T'] = max_T

            for p in fitting:
                if p.count('(')==0:
                    pars[p] = best[p]

            # copy any group-specific parameters
            for p in fitting:
                if p.count('(%s)' % grp)==1:
                    pars[p.rstrip('(%s)' % grp)] = best[p]

            # run the model for each problem
            for pid in problems:
                pars['pid'] = pid
                results[(grp,pid)] = model(problems[pid], pars)

        return results


def pred_quantiles(pred, quantiles=[.25, .5, .75]):
    dist = np.sum([pred['p_resp'][i]*pred['p_stop_cond'][:,i] for i in [0,1]], axis=0)
    return np.array([np.sum(np.cumsum(dist) < q) for q in quantiles])


def pred_quantiles_all(pred, quantiles=[.25, .5, .75]):
    arr = []
    for probid in pred.keys():
        dist = np.sum([pred[probid]['p_resp'][i]*pred[probid]['p_stop_cond'][:,i] for i in [0,1]], axis=0)
        arr.append(np.array([np.sum(np.cumsum(dist) < q) for q in quantiles]))
    return np.mean(arr, axis=0)
