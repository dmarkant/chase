import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


def checkpath(dir):
    head, tail = os.path.split(dir)
    if head!="" and head!="~" and not os.path.exists(head): checkpath(head)
    if not os.path.exists(dir): os.mkdir(dir)


def sim_id_str(name, fixed, par):
    """Get a model label given fixed and fit parameter lists"""
    id = "%s(" % name
    if par!={}:
        keys = par.keys()
        keys.sort()
        for k in keys:
            id += "%s,"%k
        id = id.rstrip(",")+"|"

    if fixed!=None:
        keys = fixed.keys()
        keys.sort()
        for k in keys:
            id += "%s=%s," % (k, fixed[k])
        id = id.rstrip(",")
    id += ")"
    return id


def unpack(value, args):
    verbose = args.get('verbose', False)
    pars = deepcopy(args)
    fitting = pars['fitting']
    if verbose: print 'evaluating:'
    for i, k in enumerate(fitting):
        pars[k] = value[i]
        if verbose: print '  %s=%s' % (k, pars[k])

    return pars, fitting, verbose


def outside_bounds(value, fitting):
    outside = False
    for v, p in zip(value, fitting.keys()):
        v_min, v_max = fitting[p]
        if v < v_min or v > v_max:
            outside = True
    return outside


def bic(negllh, n_free, n_obs):
    return 2 * negllh + n_free * np.log(n_obs)


def pfix(p):
    return np.min([np.max([p, 1e-10]), 1])


def pfixa(arr):
    arr[arr < 1e-10] = 1e-10
    return arr


def pred_quantiles(p_stop, quantiles=[.25, .5, .75]):
    return np.array([np.sum(np.cumsum(p_stop) < q) for q in quantiles]) + 1


def plot_result(result):
    fig, ax = plt.subplots(figsize=(5, 3))

    for i, label in enumerate(labels):
        ax[i].plot(result_baseline[label]['p_stop_cond'][:,1], styl[0], label='objective', color='black')
        ax[i].plot(result_value[label]['p_stop_cond'][:,1], styl[1], label='utility weighting', color='black')
        ax[i].plot(result_prob[label]['p_stop_cond'][:,1], styl[2], label='probability weighting', color='black')

    for axi in ax:
        axi.legend()
        axi.set_xlim(0, 50)
        axi.set_xlabel('Sample size')

    ax[0].set_ylabel('p(sample size|H)')

    plt.tight_layout()
    plt.show()
