import pandas as pd
from copy import deepcopy
from utils import *
import pickle
import numpy as np
from fit_process import *
from process_model import *


DATADIR = 'data'


def get_id(gen_model, problems_id, fit_model=None):
    name = '%s_recovery_%s' % (gen_model, problems_id)
    if fit_model == None:
        return name
    else:
        return '%s_%s' % (name, fit_model)


def get_N(problems):
    """Maintain a constant dataset size across different
    problem sets."""
    return int(np.floor(1200/float(len(problems))))


def simulate_data_pth(name, pars, iteration=None):
    SIM_ID = sim_id_str(name, pars, {})
    savepath = '%s/%s' % (DATADIR, name)
    checkpath(savepath)
    if iteration is None:
        return '%s/%s.csv' % (savepath, SIM_ID)
    return '%s/%s_iteration=%s.csv' % (savepath, SIM_ID, iteration)


def simulate_data(name, model, problems, pars,
                  iteration=None, problemtype='multinomial', relfreq=False):
    """
    Generate simulated data from CHASE process model.

    """
    pth = simulate_data_pth(name, pars, iteration=iteration)

    results = {}
    for pid in problems:
        ppars = deepcopy(pars)
        ppars['probid'] = pid

        # run model for this problem
        results[pid] = model(problems[pid], ppars, trackobs=relfreq)

    # create dataset
    N = pars.get('N', 100)
    data = []
    for pid in problems:
        arr = np.transpose((range(N),
                            [pid for _ in range(N)],
                            np.zeros(N, int),
                            np.array(results[pid]['choice'], int),
                            np.array(results[pid]['samplesize'], int)))
        data.append(arr)
    data = pd.DataFrame(np.concatenate(data),
                        columns=['subject', 'problem', 'group', 'choice', 'samplesize'])
    data.loc[:,'choice'] = np.array(data['choice'].values, int)
    data.loc[:,'samplesize'] = np.array(data['samplesize'].values, int)


    # get relative frequencies
    if relfreq:

        nout = 2
        cols = [['Lx%s' % i, 'Lp%s' % i, 'Lf%s' % i] for i in range(nout)] + \
               [['Hx%s' % i, 'Hp%s' % i, 'Hf%s' % i] for i in range(nout)]
        cols = np.ravel(cols)
        for c in cols:
            data[c] = np.nan

        for i, row in data.iterrows():

            pid = int(row['problem'])
            sid = int(row['subject'])
            p = problems[pid]

            sampled = results[pid]['sampled_option'][sid]
            outcome_ind = results[pid]['outcome_ind'][sid]

            f = np.zeros((2,2), int)
            for option_i in range(p.shape[0]):

                opt = p[option_i]

                for outcome_i in range(opt.shape[0]):
                    f[option_i, outcome_i] = len(outcome_ind[(sampled==option_i) & (outcome_ind==outcome_i)])

            try:
                assert np.sum(f[0]==0) < 2
                assert np.sum(f[1]==0) < 2
            except:
                print sampled
                print outcome_ind
                print f
                print

            v = []
            for option_i in range(2):
                for outcome_i in range(2):
                    v = v + [p[option_i,outcome_i,0],
                             p[option_i,outcome_i,1],
                             f[option_i,outcome_i]/float(f[option_i].sum())]

            data.loc[i,cols] = np.round(v, 3)


    print 'saving simulated data to: %s' % pth
    data.to_csv(pth)


def load_problem_set(problems_id):

    problems = {}

    if problems_id == 'sixproblems':

        arr = np.genfromtxt('data/six_problems.csv', delimiter=',')
        for i in range(len(arr)):
            problems[i] = arr[i].reshape((2,2,2))


    elif problems_id == 'glockner':

        def problem_array(row):
            return np.array([[row[['g1o1', 'g1p1']].values, row[['g1o2', 'g1p2']].values],
                             [row[['g2o1', 'g2p1']].values, row[['g2o2', 'g2p2']].values]])

        data = pd.read_csv('data/glockner2016/Glockner2016_EB_all.csv', index_col=0)
        data = data[(data.description==0) & (data.exp==3)]
        probdf = data[['g1o1', 'g1p1', 'g1o2', 'g1p2', 'g2o1', 'g2p1', 'g2o2', 'g2p2']].drop_duplicates().reset_index()

        for i, row in probdf.iterrows():
            p = problem_array(row)
            evA = expected_value(p[0])
            evB = expected_value(p[1])
            if evA > evB:
                problems[i] = np.array([p[1], p[0]])
            else:
                problems[i] = p


    elif problems_id == 'tpt':

        with open('/Users/markant/code/chase/chase/paper/data/tpt/tpt_problems.pkl', 'r') as f:
            problems = pickle.load(f)


    return problems


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
        #result = cpt.fit_msd(problems, data, name, fixed=fixed, fitting=fitting,
        #                     niter=fit_iterations, outdir=name)
        result = cpt.fit(problems, data, name, fixed=fixed, fitting=fitting,
                         niter=fit_iterations, outdir=name)


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


