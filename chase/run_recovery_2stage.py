import numpy as np
import pandas as pd
from process_model_2stage import *
from recovery import *
import itertools
from process2_recovery import fit_simulated_data, fit_simulated_data_cpt
from fit_process import best_result

Ts = [2, 4, 6, 8, 10, 50, 100]
gammas = [.6, 1, 1.4]
gen_s = 2


def generate_data(problems_id, problems):

    print '[%s] generating data with TwoStageCPTModel' % problems_id

    model = TwoStageCPTModel(stopdist='fixed-T')

    name = 'cpt_recovery_' + problems_id
    N = get_N(problems)

    for stop_T, prelec_gamma in itertools.product(Ts, gammas):

        print stop_T, prelec_gamma

        gen_pars = {'N': N,
                    'stop_T': stop_T,
                    's': gen_s,
                    'prelec_gamma': prelec_gamma,
                    'minsamplesize': 2}

        simulate_data(name, model, problems,
                      gen_pars,
                      problemtype='multinomial', relfreq=True)


def fit_cpt(problems_id, problems, relfreq=False, force=False):

    name = 'cpt_recovery_' + problems_id
    N_FIT_ITER = 2
    N = get_N(problems)

    fixed = {'gen_s': gen_s}
    fitting = ['s', 'prelec_gamma']

    for stop_T, prelec_gamma in itertools.product(Ts, gammas):

        gen_pars = {'N': N,
                    's': gen_s,
                    'stop_T': stop_T,
                    'prelec_gamma': prelec_gamma,
                    'minsamplesize': 2}

        fixed['stop_T'] = stop_T
        fixed['gen_prelec_gamma'] = prelec_gamma
        print stop_T, prelec_gamma
        data, best, pred = fit_simulated_data_cpt(name, problems, gen_pars, fitting, fixed,
                                                  iterations=N_FIT_ITER, relfreq=relfreq, force=force)


def fit_chase(problems_id, problems, stoprule='optional', force=False, N_FIT_ITER=3):

    gen_name = get_id('cpt', problems_id)
    if stoprule=='optional':
        fixed = {'gen_s': gen_s, 'minsamplesize': 2}
        fitting = ['theta', 'prelec_gamma']
    else:
        fixed = {'gen_s': gen_s}
        fitting = ['prelec_gamma']

    N = get_N(problems)

    for stop_T, prelec_gamma in itertools.product(Ts, gammas):


        gen_pars = {'N': N,
                    's': gen_s,
                    'stop_T': stop_T,
                    'prelec_gamma': prelec_gamma,
                    'minsamplesize': 2}

        fixed['stop_T'] = stop_T
        fixed['gen_prelec_gamma'] = prelec_gamma
        print stop_T, prelec_gamma


        data, best, pred = fit_simulated_data(gen_name, problems, gen_pars, fitting, fixed,
                                              stoprule=stoprule, force=force)


def compile(problems_id, fit_id):

    results = pd.DataFrame(columns=['problem', 'iteration', 'gen_N',
                                    'gen_cp', 'gen_ss_mn',
                                    'gen_stop_T', 'gen_s', 'gen_prelec_gamma',
                                    'bf_cp', 'bf_ss_mn', 'bf_s', 'bf_prelec_gamma'])

    name = get_id('cpt', problems_id, fit_model=fit_id)


    if fit_id.count('cpt') > 0:
        fitting = ['s', 'prelec_gamma']
        fixed = {'gen_s': gen_s}
        opt = 'msd'
        SIM_ID = sim_id_str(name, fixed, {f: None for f in fitting})
        print SIM_ID


    elif fit_id.count('chase_fixedT') > 0:
        fitting = ['prelec_gamma']
        fixed = {'gen_s': gen_s}
        opt = 'nllh'
        SIM_ID = sim_id_str(name, fixed, {f: None for f in fitting})
        print SIM_ID


    for stop_T, prelec_gamma in itertools.product(Ts, gammas):
        ind = results.shape[0]
        results.loc[ind] = np.nan
        results.loc[ind,'gen_s'] = gen_s
        results.loc[ind,'gen_stop_T'] = stop_T
        results.loc[ind,'gen_prelec_gamma'] = prelec_gamma

        fixed['stop_T'] = stop_T
        fixed['gen_prelec_gamma'] = prelec_gamma
        best = best_result(name, fixed, fitting, outdir=name, opt=opt)

        if best is not None:
            for p in fitting:
                results.loc[ind,'bf_%s' % p] = best[p]

    results.to_csv('%s/%s_recovery_results.csv' % (name, SIM_ID))



if __name__=='__main__':

    for problems_id in ['sixproblems', 'tpt', 'glockner']:
        problems = load_problem_set(problems_id)

        #generate_data(problems_id, problems)
        #fit_cpt(problems_id, problems, relfreq=False, force=True)
        #fit_cpt(problems_id, problems, relfreq=True, force=True)

        # fit chase-fixed
        fit_chase(problems_id, problems, stoprule='fixedT', force=True)

        # fit chase-optional

        #compile(problems_id, 'cpt')
        #compile(problems_id, 'cpt_rf')
        compile(problems_id, 'chase_fixedT')
