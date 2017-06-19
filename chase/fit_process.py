import os
from utils import checkpath, sim_id_str, bic
import pandas as pd
import numpy as np
from copy import deepcopy
from random import uniform
from scipy.optimize import minimize, fmin, differential_evolution
from collections import OrderedDict
from scipy.stats.mstats import mquantiles


PARS = {'theta': [.1, 300],
        'p_stay': [0, 1, .1],
        'p_stop_geom': [0, 1],
        'tau': [.00001, 50],
        'tau_unif': [.001, 300],
        'tau_unif_rel': [.001, 1],
        'prelec_gamma': [.01, 3],
        'prelec_elevation': [.01, 3],
        'prelec_gamma_loss': [.01, 3],
        'prelec_elevation_loss': [.01, 3],
        'pow_gain': [.01, 1.5],
        'w_loss': [0., 10],
        's': [0, 30],
        'sc': [0, 1],
        'r': [0, .1],
        'c': [0, 100],
        'c_0': [0, 100],
        'c_sigma': [0, 100]}


def freepars(parset, PARS):
    fitting = {}
    for p in parset:
        if p.count('(') > 0:
            fitting[p] = PARS[p.split('(')[0]]
        else:
            fitting[p] = PARS[p]
    return fitting


def fit_mlh(model, problems, data, name,
            fixed={}, fitting=[], niter=5, ftol=.001,
            outdir='.', method='Nelder-Mead', save=True, quiet=False):
    """Use maximum likelihood to fit CHASE model"""
    fitting = freepars(fitting, PARS)
    sim_id = sim_id_str(name, fixed, fitting)
    checkpath(outdir)



    # determine number of parameters and observations
    k = len(fitting)
    N = data.shape[0]

    # create fit table
    cols = ['iteration', 'success', 'nllh', 'k', 'N', 'bic']
    rest = fitting.keys()
    rest.sort()
    cols += rest
    arr = []
    for i in range(niter):
        arr.append([i, np.nan, np.nan, k, N, np.nan] + [np.nan for _ in range(k)])
    fitdf = pd.DataFrame(arr, columns=cols)

    # iterate through
    for i, row in fitdf.iterrows():

        # update pars with current values
        pars = deepcopy(fixed)
        pars['fitting'] = OrderedDict([(p, fitting[p]) for p in rest])
        allpars = pars['fitting'].keys() + fixed.keys()
        pars['nonspec'] = filter(lambda k: k.count('(')==0, allpars)
        pars['spec'] = filter(lambda k: k.count('(') > 0, allpars)
        print '\t'.join(pars['fitting'])

        init = []
        bounds = []
        for p in rest:
            if p=='mu':
                init.append(data.samplesize.mean())
            elif len(fitting[p]) == 3:
                init.append(fitting[p][2])
            else:
                init.append(uniform(fitting[p][0], fitting[p][1]))
            bounds.append((fitting[p][0], fitting[p][1]))

        #f = minimize(model.nloglik_opt, init, (problems, data, pars,),
        #             method=method, options={'ftol': ftol})

        f = differential_evolution(model.nloglik_opt, bounds,
                                   args=(problems, data, pars,),
                                   disp=True, polish=False)

        fitdf.ix[i,'success'] = f['success']
        fitdf.ix[i,'nllh'] = f['fun']
        fitdf.ix[i,'bic'] = bic(f['fun'], k, N)
        for v, p in enumerate(pars['fitting'].keys()):
            fitdf.ix[i,p] = f['x'][v]

        if not quiet:
            print sim_id
            print '%s/%s' % (i, fitdf.shape[0])
            print fitdf.ix[i]

    # save the table
    if save:
        print '%s/%s.csv' % (outdir, sim_id)
        fitdf.to_csv('%s/%s.csv' % (outdir, sim_id))
    return fitdf


def load_results(name, fixed={}, fitting=[], outdir='.'):
    fitting = freepars(fitting, PARS)
    sim_id = sim_id_str(name, fixed, fitting)
    pth = '%s/%s.csv' % (outdir, sim_id)
    if os.path.exists(pth):
        return pd.read_csv('%s/%s.csv' % (outdir, sim_id))
    else:
        return []


def best_result(name, fixed={}, fitting=[], outdir='.', nopars=False, opt='nllh'):
    fitdf = load_results(name, fixed=fixed, fitting=fitting, outdir=outdir)
    #fitdf = fitdf[fitdf.success==True].sort('nllh').reset_index()
    fitting = freepars(fitting, PARS)
    sim_id = sim_id_str(name, fixed, fitting)

    if len(fitdf)==0 or opt not in fitdf.columns:
        return None
    else:
        fitdf = fitdf.sort_values(by=opt).reset_index()
        fitdf['sim_id'] = sim_id
        if fitdf.shape[0] == 0:
            return pd.Series({'sim_id': sim_id})
        if nopars:
            r = fitdf.ix[0,['sim_id', 'nllh', 'k', 'N', 'bic']]
            return r
        else:
            return fitdf.ix[0]


def predict_from_result(model, problems, data, name, fixed={}, fitting=[], groups=None, outdir='.',
                        max_T=1000, N=1000):

    # load the best result
    best = best_result(name, fixed, fitting, outdir=outdir)
    fitting = freepars(fitting, PARS)

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

    knownobs = fixed.get('knownobs', False)
    if not knownobs:

        arr = []
        for i, grp in data.groupby(['problem'] + factors):

            pars = deepcopy(fixed)
            pars['N'] = N
            pars['max_T'] = max_T
            pid = grp.iloc[0]['problem']
            pars['probid'] = pid

            for k in nonspec:
                pars[k] = best[k]
            for k in spec:
                p = k.split('(')[0]
                pars[p] = grp.iloc[0][p]

            # run the model
            r = model(problems[pid], pars)
            cp = r['choice'].mean()
            ss = mquantiles(r['samplesize'])

            # store predictions
            data.loc[grp.index,'pred_cp']      = np.round(cp, 3)
            data.loc[grp.index,'pred_ss(.25)'] = ss[0]
            data.loc[grp.index,'pred_ss(.5)'] = ss[1]
            data.loc[grp.index,'pred_ss(.75)'] = ss[2]

        return data

    else:

        for i, grp in data.groupby(['subject', 'problem']):
            pars = deepcopy(fixed)
            pars['N'] = N
            pars['max_T'] = max_T
            sid = grp.iloc[0]['subject']
            pid = grp.iloc[0]['problem']
            pars['probid'] = pid

            for k in nonspec:
                pars[k] = best[k]
            for k in spec:
                p = k.split('(')[0]
                pars[p] = grp.iloc[0][p]

            results = model(problems[pid], pars, obs=grp)

            choice = grp.choice.values[-1]
            p_choice = np.array([results['p_stop_choose_A'][-1], results['p_stop_choose_B'][-1]])[choice]
            data.loc[(data.subject==sid) & (data.problem==pid),'bf_cp'] = p_choice

        return data


def pred_quantiles(pred, quantiles=[.25, .5, .75]):
    dist = np.sum([pred['p_resp'][i]*pred['p_stop_cond'][:,i] for i in [0,1]], axis=0)
    return np.array([np.sum(np.cumsum(dist) <= q) for q in quantiles])


def pred_quantiles_all(pred, quantiles=[.25, .5, .75]):
    arr = []
    for probid in pred.keys():
        dist = np.sum([pred[probid]['p_resp'][i]*pred[probid]['p_stop_cond'][:,i] for i in [0,1]], axis=0)
        arr.append(np.array([np.sum(np.cumsum(dist) <= q) for q in quantiles]))
    return np.mean(arr, axis=0)
