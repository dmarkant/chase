import numpy as np
import pandas as pd


def value_fnc(outcomes, pars):
    """Value weighting function

    pow_gain: exponent for gains (default=1)
    pow_loss: exponent for losses (default=pow_gain)
    w_loss: loss scaling parameter (default=1)
    """
    pow_gain = pars.get('pow_gain', 1.)
    pow_loss = pars.get('pow_loss', pow_gain)
    w_loss   = pars.get('w_loss', 1.)
    gain     = (outcomes * (outcomes >= 0.)) ** pow_gain
    loss     = -w_loss * ((-1 * (outcomes * (outcomes < 0.))) ** pow_loss)
    return gain + loss


def pweight_prelec(option, pars):
    prelec_elevation = pars.get('prelec_elevation', 1.)
    prelec_gamma = pars.get('prelec_gamma', 1.)
    gaindf = None
    lossdf = None

    def w(p):
        return np.exp(-prelec_elevation * ((-np.log(p)) ** prelec_gamma))

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
                v = w(row['pr'])
                gaindf.ix[i, 'w'] = v
            else:
                v = w(gaindf.iloc[i:]['pr'].sum()) - w(gaindf.iloc[(i+1):]['pr'].sum())
                gaindf.ix[i,'w'] = v


    if len(losses) > 0:
        lossdf = pd.DataFrame(np.array(losses), columns=['id', 'outcome', 'pr', 'w']).sort('outcome').reset_index()

        for i, row in lossdf.iterrows():
            if i == 0:
                row['w'] = w(row['pr'])
                lossdf.ix[i,'w'] = w(row['pr'])
            else:
                lossdf.ix[i,'w'] = w(lossdf.iloc[:(i+1)]['pr'].sum()) - w(lossdf.iloc[:i]['pr'].sum())


    weights = np.zeros(len(option))
    for i in range(len(gains)):
        weights[gaindf.iloc[i]['id']] = gaindf.iloc[i]['w']

    for i in range(len(losses)):
        weights[lossdf.iloc[i]['id']] = lossdf.iloc[i]['w']

    return weights
