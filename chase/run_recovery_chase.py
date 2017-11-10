"""
CHASE recovery study.

- generate data with CHASE process model
- fit simulated datasets with CHASE or TwoStage CPT

"""
import numpy as np
import pandas as pd
import itertools
from process_model import CHASEProcessModel
from fit_process import best_result
from recovery import *

N = 100
#N_ITER = 50
N_ITER = 10

gen_pars = {'N': N,
            'theta': None,
            'prelec_gamma': None,
            'minsamplesize': 2,
            'switchfirst': True}


thetas = [1, 2, 3, 4, 5, 7, 9, 15]
gammas = [.6, .8, 1, 1.2, 1.4]

thetas = [1, 5, 15]
gammas = [.8, 1, 1.2]


def generate_data(problems_id, problems, iteration=None):

    print '[%s] generating data with CHASEProcessModel' % problems_id

    model = CHASEProcessModel(problems=problems,
                              startdist='indifferent')

    gen_name = get_id('process', problems_id)

    for theta, prelec_gamma in itertools.product(thetas, gammas):
        print theta, prelec_gamma
        gen_pars.update({'theta': theta, 'prelec_gamma': prelec_gamma})
        simulate_data(gen_name, model, problems, gen_pars,
                      iteration=iteration, relfreq=True)


def fit_cpt(problems_id, problems, relfreq=False, force=False,
            iteration=None, N_FIT_ITER=2):

    gen_name = get_id('process', problems_id)
    if relfreq:
        fit_name = get_id('process', problems_id, 'cpt_rf')
    else:
        fit_name = get_id('process', problems_id, 'cpt')

    #N = get_N(problems)
    fixed = {}
    if iteration != None:
        fixed['iteration'] = iteration
    fitting = ['s', 'prelec_gamma']
    #fitting = ['s']

    for theta, prelec_gamma in itertools.product(thetas, gammas):
        print theta, prelec_gamma
        gen_pars.update({'theta': theta, 'prelec_gamma': prelec_gamma})
        fixed['gen_theta'] = theta
        fixed['gen_prelec_gamma'] = prelec_gamma


        data, best, pred = fit_simulated_data_cpt(gen_name, problems, gen_pars,
                                                  fitting, fixed,
                                                  iteration=iteration,
                                                  fit_iterations=N_FIT_ITER,
                                                  relfreq=relfreq, force=force)



def fit_chase(problems_id, problems, optional=True, force=False, N_FIT_ITER=3):

    gen_name = get_id('process', problems_id)
    if optional:
        fit_name = get_id('process', problems_id, 'chase')
    else:
        fit_name = get_id('process', problems_id, 'chase_fixedT')

    N = get_N(problems)
    fixed = {'minsamplesize': 2}
    fitting = ['theta', 'prelec_gamma']
    #SIM_ID = sim_id_str(fit_name, fixed, {f: None for f in fitting})
    #print SIM_ID

    #results = pd.DataFrame(columns=['problem', 'iteration', 'gen_N',
    #                                'gen_cp', 'gen_ss_mn',
    #                                'gen_stop_T', 'gen_s', 'gen_prelec_gamma',
    #                                'bf_cp', 'bf_ss_mn', 'bf_s', 'bf_prelec_gamma'])

    for theta, prelec_gamma in itertools.product(thetas, gammas):

        gen_pars = {'N': N,
                    'theta': theta,
                    'prelec_gamma': prelec_gamma,
                    'minsamplesize': 2,}
        fixed['gen_theta'] = theta
        fixed['gen_prelec_gamma'] = prelec_gamma

        print theta, prelec_gamma

        data, best, pred = fit_simulated_data(gen_name, problems, gen_pars, fitting, fixed)

        #ind = results.shape[0]
        #results.loc[ind] = np.nan
        #results.loc[ind,'gen_theta'] = theta
        #results.loc[ind,'gen_prelec_gamma'] = prelec_gamma
        #for p in fitting:
        #    results.loc[ind,'bf_%s' % p] = best[p]



    #results.to_csv('%s/%s_recovery_results.csv' % (fit_name, SIM_ID))


def compile(problems_id, fit_id):

    name = get_id('process', problems_id, fit_model=fit_id)

    results = pd.DataFrame(columns=['problem', 'iteration', 'gen_N',
                                    'gen_cp', 'gen_ss_mn',
                                    'gen_theta', 'gen_prelec_gamma',
                                    'bf_cp', 'bf_ss_mn', 'bf_theta', 'bf_s', 'bf_prelec_gamma'])

    if fit_id.count('cpt') > 0:
        fitting = ['s', 'prelec_gamma']
        fixed = {'N': N}
        SIM_ID = sim_id_str(name, fixed, {f: None for f in fitting})
        print SIM_ID

        for iteration in range(N_ITER):

            fixed = {'iteration': iteration}
            best = best_result(name, fixed, fitting, outdir=name, opt='msd')

            for theta, prelec_gamma in itertools.product(thetas, gammas):
                ind = results.shape[0]
                results.loc[ind] = np.nan
                results.loc[ind,'iteration'] = iteration
                results.loc[ind,'gen_theta'] = theta
                results.loc[ind,'gen_prelec_gamma'] = prelec_gamma

                fixed['gen_theta'] = theta
                fixed['gen_prelec_gamma'] = prelec_gamma
                best = best_result(name, fixed, fitting, outdir=name, opt='msd')

                if best is not None:
                    for p in fitting:
                        results.loc[ind,'bf_%s' % p] = best[p]

        results.to_csv('%s/%s_recovery_results.csv' % (name, SIM_ID))


    elif fit_id.count('chase') > 0:
        fitting = ['theta', 'prelec_gamma']
        fixed = {'minsamplesize': 2}
        SIM_ID = sim_id_str(name, fixed, {f: None for f in fitting})
        print SIM_ID


        for theta, prelec_gamma in itertools.product(thetas, gammas):
            ind = results.shape[0]
            results.loc[ind] = np.nan
            results.loc[ind,'gen_theta'] = theta
            results.loc[ind,'gen_prelec_gamma'] = prelec_gamma

            fixed['gen_theta'] = theta
            fixed['gen_prelec_gamma'] = prelec_gamma
            best = best_result(name, fixed, fitting, outdir=name)
            if best is not None:
                for p in fitting:
                    results.loc[ind,'bf_%s' % p] = best[p]

    results.to_csv('%s/%s_recovery_results.csv' % (name, SIM_ID))



if __name__=='__main__':

    if True:
        for iteration in range(N_ITER):

            #for problems_id in ['sixproblems', 'tpt', 'glockner']:
            for problems_id in ['glockner']:
                problems = load_problem_set(problems_id)

                generate_data(problems_id, problems, iteration=iteration)
                fit_cpt(problems_id, problems, relfreq=False, force=True, iteration=iteration, N_FIT_ITER=3)
                #fit_cpt(problems_id, problems, relfreq=True, force=True, iteration=iteration, N_FIT_ITER=2)
                #fit_chase(problems_id, problems, force=True, N_FIT_ITER=1)

    if True:
        for problems_id in ['glockner']:
            compile(problems_id, 'cpt')
            #compile(problems_id, 'cpt_rf')
