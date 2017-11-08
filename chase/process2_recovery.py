import numpy as np
import pandas as pd
#from chase.base import *
#from chase.utils import *
#from chase.fit import *
from process_model import *
from fit_process import *
from recovery import *


def fit_simulated_data(name, problems, gen_pars, fitting, fixed,
                       stoprule='optional',
                       problemtype='multinomial', iterations=1, force=False):

    data_pth = simulate_data_pth(name, gen_pars)
    data = pd.read_csv(data_pth, index_col=0)
    print '[%s] loaded dataset from %s' % (name, data_pth)


    if stoprule=='optional':
        name = name + '_chase'
    elif stoprule=='fixedT':
        name = name + '_chase_fixedT'

    fitting = {p: PARS[p] for p in fitting}
    SIM_ID = sim_id_str(name, fixed, {f: None for f in fitting})


    # initialize the model
    m = CHASEProcessModel(problems=problems,
                          startdist='indifferent',
                          stoprule=stoprule)

    best = best_result(name, fixed, fitting.keys(), outdir=name)
    if best is None or force:

        # fit
        results = fit_mlh(m, problems, data,
                          name, fixed, fitting, niter=iterations, outdir=name, quiet=False,
                          ftol=.05)

        best = best_result(name, fixed, fitting.keys(), outdir=name)

    #pred = predict_from_result(m, problems, name,
    #                           fixed=fixed,
    #                           fitting=fitting,
    #                           outdir=name)
    pred = None
    print best
    return data, best, pred


def fit_simulated_data_cpt(name, problems, gen_pars, fitting, fixed,
                           iteration=None, fit_iterations=1, relfreq=False, force=False):

    data_pth = simulate_data_pth(name, gen_pars, iteration=iteration)
    pfx = name if iteration == None else ('%s %s' % (name, iteration))
    print '[%s] loading data from %s' % (pfx, data_pth)
    data = pd.read_csv(data_pth, index_col=0)

    if relfreq:

        problems_exp = {}

        data['problem'] = np.arange(data.shape[0], dtype=int)
        col_L = [['Lx%s' % i, 'Lf%s' % i] for i in range(2)]
        col_H = [['Hx%s' % i, 'Hf%s' % i] for i in range(2)]

        for i, row in data.iterrows():
            pid = int(row['problem'])
            problems_exp[pid] = np.array([[row.loc[col_L[j]].values for j in range(2)],
                                          [row.loc[col_H[j]].values for j in range(2)]])
        problems = problems_exp

    name = name + '_cpt'
    if relfreq: name = name + '_rf'
    fitting = {p: PARS[p] for p in fitting}
    SIM_ID = sim_id_str(name, fixed, {f: None for f in fitting})

    #best = best_result(name, fixed, fitting, outdir=name, opt='msd')
    best = best_result(name, fixed, fitting, outdir=name)
    if best is None or force:

        # initialize the model
        result = cpt.fit_msd(problems, data, name, fixed=fixed, fitting=fitting,
                             niter=fit_iterations, outdir=name)
        #result = cpt.fit(problems, data, name, fixed=fixed, fitting=fitting,
        #                 niter=fit_iterations, outdir=name)


        # fit
        best = best_result(name, fixed, fitting, outdir=name, opt='msd')
        #best = best_result(name, fixed, fitting, outdir=name)
        #pred = predict_from_result(m, problems, data, SIM_ID, fixed,
        #                           fitting = fitting,
        #                           outdir=name)
    pred = None
    print best
    return data, best, pred


def sim_and_fit(problems, gen_pars, fit_pars, problemtype='multinomial',
                pref_units='sums', gen_drift='ev', fit_drift='ev'):

    samplesize, choices = simulate_process(problems, gen_pars, problemtype=problemtype,
                                           pref_units=pref_units, drift=gen_drift)

    N = gen_pars.get('N', 1000)
    data = []
    for pid in problems:
        arr = np.transpose((np.ones(N, int)*np.nan, [pid for _ in range(N)], np.zeros(N, int),
                            np.array(choices[pid], int), np.array(samplesize[pid], int)))
        data.append(arr)


    data = pd.DataFrame(np.concatenate(data), columns=['subject', 'problem', 'group', 'choice', 'samplesize'])
    data.loc[:,'choice'] = np.array(data['choice'].values, int)
    data.loc[:,'samplesize'] = np.array(data['samplesize'].values, int)

    FIXED = {'optiontype': problemtype, 'pref_units': 'sums'}
    N_ITER = 1
    SIM_ID = 'recover_process_data'

    fitting = {p: PARS[p] for p in fit_pars}

    # initialize the model
    m = CHASEModel(drift=fit_drift,
                   startdist='laplace',
                   problems=problems,
                   problemtype=problemtype)

    # fit
    results = fit_mlh(m, problems, data,
                      SIM_ID, FIXED, fitting, niter=N_ITER, outdir=OUTDIR, quiet=True)
    best = best_result(SIM_ID, FIXED, fitting, outdir=OUTDIR)
    pred = predict_from_result(m, problems, data, SIM_ID, FIXED,
                               fitting = fitting,
                               outdir=OUTDIR)
    print best
    return data, best, pred


