from process_model import *
from fit_process import *
import pickle

N_ITER = 1

PARSETS = {}
PARSETS['optional'] = [#['theta(cost=low)', 'theta(cost=high)', 'tau', 'c'],
                       #['theta(cost=low)', 'theta(cost=high)', 'tau_unif', 'c'],
                       ['theta(cost=low)', 'theta(cost=high)', 'tau_unif', 'c_sigma'],
                       #['theta(cost=low)', 'theta(cost=high)', 'tau_unif', 'c', 'pow_gain'],
                       #['theta(cost=low)', 'theta(cost=high)', 'tau', 'c', 'r'],
                       ]

PARSETS['geometric'] = [#['p_stop_geom(cost=low)', 'p_stop_geom(cost=high)', 'tau', 'c'],
                        ['p_stop_geom(cost=low)', 'p_stop_geom(cost=high)', 'tau_unif', 'c_sigma']
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

    if stoprule is 'optional':
        SIM_ID = 'process_markant_individual_subj%s' % sid
        OUTDIR = 'process_fitresults_markant_individual'
    elif stoprule is 'geometric':
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

    # run with higher theta
    #array([ 91,  92,  93,  94,  96,  97,  98,  99, 100, 101, 102, 103, 104,
    #   105, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117, 119,
    #   120, 121, 123, 127, 128, 129, 130, 134, 135, 136, 137, 138, 139,
    #   140, 141, 142, 143, 144, 145, 146, 148, 149, 150, 151, 152, 153,
    #   154, 155, 158, 159, 161, 162, 163, 166, 169, 170, 173, 174, 175,
    #   176, 177, 178, 179, 180, 181, 183, 184, 186, 188, 190, 191, 192,
    #   194, 195, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208,
    #   209, 210, 211, 213, 214, 215, 216, 217, 218])
    #for sid in SSET:
    for sid in [111, 112, 113]:
        for stoprule in ['optional', 'geometric']:
        #for stoprule in ['geometric']:
            fit(sid, data[data.subject==sid], stoprule)


if __name__=='__main__':
    load_data()
    run()
