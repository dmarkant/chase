import os
import numpy as np
from copy import deepcopy


def checkpath(dir):
    """Recursively check that a path exists, and
    create if not."""
    head, tail = os.path.split(dir)
    if head!="" and head!="~" and not os.path.exists(head): checkpath(head)
    if not os.path.exists(dir): os.mkdir(dir)


def sim_id_str(name, fixed, par):
    """Get a model label given fixed and fit parameter lists.

    Returns a string:
        <name>(<list of free parameters>|<list of fixed parameters>)

    name: identifier for simulation
    fixed: fixed parameters
    par: free parameters
    """
    id = "%s(" % name
    if par!={}:
        keys = par.keys()
        keys.sort()
        for k in keys:
            id += "%s,"%k
        id = id.rstrip(",")+"|"

    if fixed!={}:
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
    """Given a list of parameters, check if any fall
    outside bounds."""
    outside = False
    for v, p in zip(value, fitting):
        v_min, v_max = fitting[p][:2]
        if v < v_min or v > v_max:
            outside = True
    return outside


def bic(negllh, n_free, n_obs):
    return 2 * negllh + n_free * np.log(n_obs)


def pfix(p):
    """Truncate probabilities close to 0 and 1"""
    return np.clip(p, 1e-5, 1 - 1e-5)


def sample_from_discrete(cp):
    r = np.random.random()
    return np.where(r < np.cumsum(cp))[0][0]


def expected_value(option):
    return np.dot(option[:,0], option[:,1])


def expected_variance(option):
    values = option[:,0]
    weights = option[:,1]
    ev = expected_value(option)
    return np.dot(weights, values ** 2) - ev ** 2
