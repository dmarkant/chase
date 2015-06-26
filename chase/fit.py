import os
from utils import checkpath, sim_id_str, bic
import pandas as pd
import numpy as np
from itertools import product
from copy import deepcopy
from random import uniform
from scipy.optimize import minimize, fmin
from collections import OrderedDict

def fit_mlh(model, problems, data, name, fixed={}, fitting={}, niter=5, outdir='.'):
    """Use maximum likelihood to fit CHASE model"""
    sim_id = sim_id_str(name, fixed, fitting)
    print sim_id
    checkpath(outdir)

    cols = ['iteration', 'success', 'nllh', 'k', 'N', 'bic']

    theta_min, theta_max = fitting['theta']
    thetas = filter(lambda k: k.count('theta') > 0, fitting.keys())
    theta_prod = map(list, list(product(range(theta_min, theta_max + 1), repeat=len(thetas))))
    cols += thetas

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

        #print '%s/%s' % (i, fitdf.shape[0])

        # update pars with current values of theta
        pars = deepcopy(fixed)
        for th in thetas:
            pars[th] = row[th]

        pars['fitting'] = OrderedDict([(p, fitting[p]) for p in rest])
        init = []
        for p in rest:
            if len(fitting[p]) == 3:
                init.append(fitting[p][2])
            else:
                init.append(uniform(fitting[p][0], fitting[p][1]))

        f = minimize(model.nloglik_opt, init, (problems, data, pars,),
                     method='Nelder-Mead', options={'ftol': .0001})

        fitdf.ix[i,'success'] = f['success']
        fitdf.ix[i,'nllh'] = f['fun']
        fitdf.ix[i,'bic'] = bic(f['fun'], k, N)
        for v, p in enumerate(pars['fitting'].keys()):
            fitdf.ix[i,p] = f['x'][v]

    # save the table
    fitdf.to_csv('%s/%s.csv' % (outdir, sim_id))

    return fitdf


def load_results(name, fixed={}, fitting={}, outdir='.'):
    sim_id = sim_id_str(name, fixed, fitting)
    fitdf = pd.read_csv('%s/%s.csv' % (outdir, sim_id))
    return fitdf


def best_result(name, fixed={}, fitting={}, outdir='.', nopars=False):
    sim_id = sim_id_str(name, fixed, fitting)
    fitdf = load_results(name, fixed=fixed, fitting=fitting, outdir=outdir)
    fitdf = fitdf[fitdf.success==True].sort('nllh').reset_index()
    if nopars:
        r = fitdf.ix[0,['nllh', 'k', 'N', 'bic']]
        r['sim_id'] = sim_id
        return r
    else:
        return fitdf.ix[0]


def predict_from_result(model, problems, name, fixed={}, fitting={}, outdir='.'):

    # load the best result
    best = best_result(name, fixed, fitting, outdir=outdir)

    # copy best-fit parameter settings
    pars = deepcopy(fixed)
    for p in fitting:
        pars[p] = best[p]

    # run the model for each problem
    results = {}
    for pid in problems:
        results[pid] = model(problems[pid], pars)

    return results


def pred_quantiles(pred, quantiles=[.25, .5, .75]):
    dist = np.sum([pred[0]['p_resp'][i]*pred[0]['p_stop_cond'][:,i] for i in [0,1]], axis=0)
    return np.array([np.sum(np.cumsum(dist) < q) for q in quantiles])
