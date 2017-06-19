from process_model import *
from fit_process import *
import pickle
from farming import *

N_ITER = 1

PARSETS = {}
PARSETS['optional'] = [#['theta(cost=low)', 'theta(cost=high)', 'tau', 'c'],
                       #['theta(cost=low)', 'theta(cost=high)', 'tau_unif', 'c'],
                       #['theta(cost=low)', 'theta(cost=high)', 'tau_unif', 'c', 'sc'],
                       ['theta(cost=low)', 'theta(cost=high)', 'tau_unif', 'c_sigma'],
                       ['theta(cost=low)', 'theta(cost=high)', 'tau_unif', 'c_0'],
                       ['theta(cost=low)', 'theta(cost=high)', 'tau_unif', 'c_0', 'sc'],
                       #['theta(cost=low)', 'theta(cost=high)', 'tau_unif_rel', 'c_0', 'sc'],
                       #['theta(cost=low)', 'theta(cost=high)', 'tau', 'c_0'],
                       #['theta(cost=low)', 'theta(cost=high)', 'tau_unif', 'c', 'pow_gain'],
                       #['theta(cost=low)', 'theta(cost=high)', 'tau', 'c', 'r'],
                       ]

PARSETS['geometric'] = [#['p_stop_geom(cost=low)', 'p_stop_geom(cost=high)', 'tau', 'c'],
                        #['p_stop_geom(cost=low)', 'p_stop_geom(cost=high)', 'tau', 'c', 'sc'],
                        ['p_stop_geom(cost=low)', 'p_stop_geom(cost=high)', 'tau_unif', 'c_sigma'],
                        ['p_stop_geom(cost=low)', 'p_stop_geom(cost=high)', 'tau_unif', 'c_0'],
                        ['p_stop_geom(cost=low)', 'p_stop_geom(cost=high)', 'tau_unif', 'c_0', 'sc']
                        #['p_stop_geom(cost=low)', 'p_stop_geom(cost=high)', 'tau_unif_rel', 'c_0', 'sc']
                        #['p_stop_geom(cost=low)', 'p_stop_geom(cost=high)', 'tau', 'c_0']
                        #['p_stop_geom(cost=low)', 'p_stop_geom(cost=high)', 'tau_unif', 'c']
                        ]

FIXED = {}
for pref_units in ['sums', 'diffs']:
    FIXED[('optional',pref_units)] = {'pref_units': pref_units,
                                      'stoprule': 'optional',
                                      'max_T': 1000,
                                      'N': 10000,}

    FIXED[('geometric',pref_units)] = {'pref_units': pref_units,
                                       'stoprule': 'fixedGeom',
                                       'N': 10000,
                                       'max_T': 1000,}



def load_data():
    global data, problems
    data = pd.read_csv('paper/data/markant/results_v3.csv', index_col=0)
    data.rename(columns={'sid': 'subject', 'probid': 'problem'}, inplace=True)
    data['choice'] = data.choice.apply(lambda c: 1 if c=='B' else 0)

    probdf = pd.read_csv('paper/data/markant/problems_normal.csv', index_col=0)
    probdf['mn_diff'] = probdf['mn_B'] - probdf['mn_A']
    probdf['variance'] = probdf['var_A'] # matched variance in both options

    problems = {}
    for i, row in probdf.iterrows():
        problems[row['id']] = np.array([[row['mn_A'], row['var_A']],
                                        [row['mn_B'], row['var_B']]])

    problems_obsvar = {}
    for pid in problems:
        obs_var = []
        mu = problems[pid][:,0]
        sigma2 = problems[pid][:,1]/2.
        problems_obsvar[pid] = np.array([[mu[0], sigma2[0]], [mu[1], sigma2[1]]])

    problems = problems_obsvar


def fit(sid, data, stoprule):

    print 'subject %s [%s]' % (sid, stoprule)

    if stoprule == 'optional':
        SIM_ID = 'process_markant_individual_subj%s' % sid
        OUTDIR = 'process_fitresults_markant_individual'
    elif stoprule == 'geometric':
        SIM_ID = 'process_planned_markant_individual_subj%s' % sid
        OUTDIR = 'process_planned_fitresults_markant_individual'

    P = PARSETS[stoprule]
    F = FIXED[(stoprule,'diffs')]

    sr = 'fixedGeom' if stoprule=='geometric' else 'optional'

    m = CHASEProcessModel(problems=problems,
                          problemtype='normal',
                          stoprule=sr,
                          startdist='laplace')

    for fitting in P:
        results = fit_mlh(m, problems, data,
                          SIM_ID, F, fitting, ftol=.1, niter=N_ITER, outdir=OUTDIR)
        print results.sort('nllh')


def best(sid, stoprule, fitting):
    if stoprule is 'optional':
        SIM_ID = 'process_markant_individual_subj%s' % sid
        OUTDIR = 'process_fitresults_markant_individual'
    elif stoprule is 'geometric':
        SIM_ID = 'process_planned_markant_individual_subj%s' % sid
        OUTDIR = 'process_planned_fitresults_markant_individual'

    F = FIXED[(stoprule,'diffs')]
    best = best_result(SIM_ID, F, fitting, outdir=OUTDIR)
    return best


def predict(sid, data, problems, stoprule, fitting):

    if stoprule is 'optional':
        SIM_ID = 'process_markant_individual_subj%s' % sid
        OUTDIR = 'process_fitresults_markant_individual'
    elif stoprule is 'geometric':
        SIM_ID = 'process_planned_markant_individual_subj%s' % sid
        OUTDIR = 'process_planned_fitresults_markant_individual'

    sr = 'fixedGeom' if stoprule=='geometric' else 'optional'
    m = CHASEProcessModel(problems=problems,
                          problemtype='normal',
                          stoprule=sr,
                          startdist='laplace')

    F = FIXED[(stoprule,'diffs')]
    #best = best_result(SIM_ID, F, fitting, outdir=OUTDIR)
    pred = predict_from_result(m, problems, data, SIM_ID, F,
                               fitting = fitting,
                               outdir=OUTDIR)
    return pred


def run():

    SSET=data.subject.unique()
    SSET = [117]

    for sid in SSET:
        for stoprule in ['optional', 'geometric']:
            fit(sid, data[data.subject==sid], stoprule)


# Run in parallel

def f(args):
    sid, stoprule = args
    fit(sid, data[data.subject==sid], stoprule)
    return 1


def run_multi():

    SSET = data.subject.unique()
    jobs = []
    for sid in SSET:
        for stoprule in ['optional', 'geometric']:
            jobs.append([sid,stoprule])

    r = farm(targetfunc=f, jobs=jobs, num_workers=14)
    print "result: ", r

    # incompleted jobs
    print "incomplete: ", catch_incomplete_jobs(r)


if __name__=='__main__':
    load_data()
    run_multi()
    #run()
