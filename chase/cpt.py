import numpy as np
import pandas as pd
from copy import deepcopy

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

    # if there are zero outcomes but no gains, group with losses
    for i, opt in enumerate(option):
        if opt[0] == 0:
            if len(gains) == 0:
                losses.append([i, opt[0], opt[1], np.nan])
            else:
                gains.append([i, opt[0], opt[1], np.nan])

    if len(gains) > 0:
        gaindf = pd.DataFrame(np.array(gains), columns=['id', 'outcome', 'pr', 'w']).sort('outcome').reset_index()
    if len(losses) > 0:
        lossdf = pd.DataFrame(np.array(losses), columns=['id', 'outcome', 'pr', 'w']).sort('outcome').reset_index()

    return gaindf, lossdf



def pweight_prelec(option, pars):
    prelec_elevation = pars.get('prelec_elevation', 1.)
    prelec_gamma = pars.get('prelec_gamma', 1.)

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
        wr = w_prelec(r, prelec_elevation, prelec_gamma)
        wrd = np.ediff1d(wr)
        lossdf.w = wrd

    # put ranked weights back in original order
    weights = np.zeros(n_gains + n_losses)
    for i in range(n_gains):
        weights[gaindf.iloc[i]['id']] = gaindf.iloc[i]['w']
    for i in range(n_losses):
        weights[lossdf.iloc[i]['id']] = lossdf.iloc[i]['w']

    assert not np.any(np.isnan(weights))

    # normalize (in case of mixed option)
    weights = weights / np.sum(weights)

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
