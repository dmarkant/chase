import numpy as np
import pandas as pd


def value_fnc(outcomes, pars):
    """Value weighting function

    pow_gain: exponent for gains (default=1)
    pow_loss: exponent for losses (default=pow_gain)
    w_loss: loss scaling parameter (default=1)
    """
    pow_gain = pars.get('pow_gain', 1.)
    w_loss   = pars.get('w_loss', 1.)
    gain     = (outcomes * (outcomes >= 0.)) ** pow_gain
    loss     = -w_loss * ((-1 * (outcomes * (outcomes < 0.))) ** pow_gain)
    return gain + loss

"""
def pweight_prelec(option, pars):
    prelec_elevation = pars.get('prelec_elevation', 1.)
    prelec_gamma = pars.get('prelec_gamma', 1.)
    prelec_elevation_loss = pars.get('prelec_elevation_loss', prelec_elevation)
    prelec_gamma_loss = pars.get('prelec_gamma', prelec_gamma)

    gaindf = None
    lossdf = None

    def w(p):
        return np.exp(-prelec_elevation * ((-np.log(p)) ** prelec_gamma))

    def w_loss(p):
        return np.exp(-prelec_elevation_loss * ((-np.log(p)) ** prelec_gamma_loss))

    gains = []
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

        for i, row in gaindf.iterrows():

            if i == (len(gaindf) - 1):
                gaindf.ix[i, 'w'] = w(row['pr'])
            else:
                v = w(gaindf.iloc[i:]['pr'].sum()) - w(gaindf.iloc[(i+1):]['pr'].sum())
                gaindf.ix[i,'w'] = v


    if len(losses) > 0:
        lossdf = pd.DataFrame(np.array(losses), columns=['id', 'outcome', 'pr', 'w']).sort('outcome').reset_index()

        for i, row in lossdf.iterrows():
            if i == 0:
                lossdf.ix[i,'w'] = w_loss(row['pr'])
            else:
                lossdf.ix[i,'w'] = w_loss(lossdf.iloc[:(i+1)]['pr'].sum()) - w_loss(lossdf.iloc[:i]['pr'].sum())


    weights = np.zeros(len(option))
    for i in range(len(gains)):
        weights[gaindf.iloc[i]['id']] = gaindf.iloc[i]['w']

    for i in range(len(losses)):
        weights[lossdf.iloc[i]['id']] = lossdf.iloc[i]['w']

    try:
        assert np.sum(np.isnan(weights))==0
    except AssertionError:
        print weights
        print gaindf
        print lossdf
        print prelec_elevation
        print prelec_gamma
        print dummy

    return weights
"""



def setup_option(option):
    gaindf = None
    lossdf = None

    gains = []
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

    gaindf, lossdf = setup_option(option)

    n_gains = gaindf.shape[0] if gaindf is not None else 0
    n_losses = lossdf.shape[0] if lossdf is not None else 0
    prelec_elevation = pars.get('prelec_elevation', 1.)
    prelec_gamma = pars.get('prelec_gamma', 1.)

    if gaindf is not None:

        q = gaindf.pr.values
        r = np.append(np.cumsum(q[::-1])[::-1], [0])
        wr = w(r, prelec_elevation, prelec_gamma)
        wr[0] = 1.
        wrd = -np.ediff1d(wr)
        gaindf.w = wrd

    if lossdf is not None:
        q = lossdf.pr.values
        r = np.append([0], np.cumsum(q))
        wr = w(r, prelec_elevation, prelec_gamma)
        wrd = np.ediff1d(wr)
        lossdf.w = wrd


    weights = np.zeros(n_gains + n_losses)
    for i in range(n_gains):
        weights[gaindf.iloc[i]['id']] = gaindf.iloc[i]['w']

    for i in range(n_losses):
        weights[lossdf.iloc[i]['id']] = lossdf.iloc[i]['w']

    try:
        assert np.sum(np.isnan(weights))==0
    except AssertionError:
        print weights
        print wr
        print wrd
        print gaindf
        print lossdf
        print prelec_elevation
        print prelec_gamma
        print dummy


    return weights




def setup(problems):
    rdw = {}
    for pid in problems:
        options = problems[pid]

        rdw[pid] = []

        for option in options:

            gaindf = None
            lossdf = None

            gains = []
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

            rdw[pid].append({'gaindf': gaindf, 'lossdf': lossdf})

    return rdw


def w(p, delta, gamma):
    return np.exp(-delta* ((-np.log(p)) ** gamma))


def pweight_prelec_known_problem(option, pars):

    gaindf = option['gaindf']
    lossdf = option['lossdf']
    n_gains = gaindf.shape[0] if gaindf is not None else 0
    n_losses = lossdf.shape[0] if lossdf is not None else 0
    prelec_elevation = pars.get('prelec_elevation', 1.)
    prelec_gamma = pars.get('prelec_gamma', 1.)

    if gaindf is not None:

        q = gaindf.pr.values
        r = np.append(np.cumsum(q[::-1])[::-1], [0])
        wr = w(r, prelec_elevation, prelec_gamma)
        wr[0] = 1.
        wrd = -np.ediff1d(wr)
        gaindf.w = wrd

    if lossdf is not None:
        q = lossdf.pr.values
        r = np.append([0], np.cumsum(q))
        wr = w(r, prelec_elevation, prelec_gamma)
        wrd = np.ediff1d(wr)
        lossdf.w = wrd


    weights = np.zeros(n_gains + n_losses)
    for i in range(n_gains):
        weights[gaindf.iloc[i]['id']] = gaindf.iloc[i]['w']

    for i in range(n_losses):
        weights[lossdf.iloc[i]['id']] = lossdf.iloc[i]['w']

    try:
        assert np.sum(np.isnan(weights))==0
    except AssertionError:
        print weights
        print gaindf
        print lossdf
        print prelec_elevation
        print prelec_gamma
        print dummy


    return weights
