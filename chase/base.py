from copy import deepcopy
import numpy as np
from scipy import linalg
from numpy.linalg import matrix_power
from drift import *
from initial_distribution import *
from stopping import TruncatedNormal, Geometric
from utils import *
from time import time


class CHASEModel(object):

    def __init__(self, **kwargs):
        self.data = kwargs.get('data', None)
        self.problems = kwargs.get('problems', None)
        self.problemtype = kwargs.get('problemtype', 'multinomial')

        # set function for drift rate
        drift = kwargs.get('drift', 'ev')
        if isinstance(drift, float) or isinstance(drift, int):
            self.drift = lambda *args, **kwargs: drift
        if drift is 'ev':
            self.drift = DriftModel()
        elif drift is 'cpt':
            self.drift = CPTDriftModel(problems=self.problems, problemtype=self.problemtype)

        # set function for initial distribution
        startdist = kwargs.get('startdist', 'indifferent')
        if 'numpy' in str(type(startdist)):
            self.Z = lambda *args, **kwargs: startdist
        elif startdist is 'uniform':
            self.Z = uniform_initial_distribution
        elif startdist is 'softmax':
            self.Z = softmax_initial_distribution
        elif startdist is 'indifferent':
            self.Z = indifferent_initial_distribution
        elif startdist is 'laplace':
            self.Z = laplace_initial_distribution


    def set_statespace(self, pars):
        self.theta = int(pars.get('theta', 5))
        self.V = np.arange(-self.theta, self.theta+1, 1, dtype=int)
        self.m = len(self.V)
        return


    def __call__(self, options, pars):
        """Evaluate the model for a given set of parameters
        and option set."""

        self.dt = pars.get('stepsize', 1) # size of increments
        self.max_T = int(pars.get('max_T', 100))    # range of timesteps

        self.theta = int(np.floor(pars.get('theta', 5))) # controlling size of state space
        self.V = np.arange(-self.theta/self.dt,
                           self.theta/self.dt+self.dt,
                           self.dt, dtype=float)
        self.m = len(self.V)
        #self.set_statespace(pars)

        vi_pqr = np.concatenate((np.array([0, self.m - 1]), np.arange(1, self.m - 1)))
        self.V_pqr = self.V[vi_pqr] # sort state space

        # evaluate the starting distribution
        if self.theta==1 and self.dt==1:
            Z = np.array([[1]])
        else:
            Z = self.Z(self.m - 2, pars)
        try:
            assert np.isclose(Z.sum(), 1)
        except:
            print pars
            print Z
            print dummy

        # transition matrix
        tm_pqr = self.transition_matrix_PQR(options, pars)
        Q = tm_pqr[2:,2:]
        I = np.eye(self.m - 2)
        R = tm_pqr[2:,:2]
        IQ = np.matrix(linalg.inv(I - Q))

        # if there was a minimum sample size, update the
        # initial distribution for that number of samples
        if 'minsamplesize' in pars:
            min_ss = pars.get('minsamplesize')
            #if min_ss == 2:
            #    Z = Z * Q
            #else:
            #    Z = Z * matrix_power(Q, int((min_ss - 1)/self.dt))
            Z = Z * matrix_power(Q, int((min_ss - 1)/self.dt))
        Z = np.matrix(Z / np.sum(Z))

        S = np.zeros((self.max_T/self.dt, self.m - 2, self.m - 2))
        S[0] = np.eye(self.m - 2)
        for i in range(1, int(self.max_T/self.dt)):
            S[i] = np.dot(S[i-1], Q)
        SR = np.tensordot(S, R, axes=1)
        states_t = np.dot(Z, S)

        # overall response probabilities
        # (see Diederich and Busemeyer, 2003, eqn. 2)
        p_resp = Z * (IQ * R)

        # response probability over timesteps
        p_resp_t = np.dot(Z, SR)

        # probability of stopping over time
        # (see Diederich and Busemeyer, 2003, eqn. 18)
        p_stop_cond = np.array([pt/p_resp for pt in p_resp_t]).reshape((self.max_T/self.dt, 2))
        p_stop_cond = p_stop_cond.reshape((self.max_T, 1./self.dt, 2)).sum(axis=1)

        # expected number of timesteps, conditional on choice
        # (see Diederich and Busemeyer, 2003, eqn. 3)
        exp_samplesize = self.dt*(Z*(IQ*IQ)*R)/p_resp

        return {'states_t': states_t,
                'p_resp': np.array(p_resp)[0],
                'p_resp_t': p_resp_t,
                'p_stop_cond': p_stop_cond,
                'exp_samplesize': exp_samplesize,
                }


    def transition_probs(self, d, pars, state=None):
        p_stay = np.min([.9999, pars.get('p_stay', 0.)])
        p_down = ((1 - p_stay)/2.) * (1 - d*np.sqrt(self.dt))
        p_up = ((1 - p_stay)/2.) * (1 + d*np.sqrt(self.dt))
        assert np.isclose(np.sum([p_down, p_stay, p_up]), 1.)
        return np.array([p_down, p_stay, p_up])


    def transition_matrix_PQR(self, options, pars, full=False):

        d = self.drift(options, pars)
        tp = np.tile(self.transition_probs(d, pars), (self.m - 2, 1))

        tm        = np.zeros((self.m, self.m), float)
        tm[0,0]   = 1.
        tm[-1,-1] = 1.

        tm_pqr = np.zeros((self.m, self.m), float)
        tm_pqr[0,0] = 1.
        tm_pqr[1,1] = 1.

        # construct PQR row by row
        V_pqr = deepcopy(self.V_pqr)
        for i in range(1, self.m - 1):
            tm[i,i-1:i+2] = tp[i-1]
            row = np.where(V_pqr==self.V[i])[0][0]
            ind_pqr = np.array([np.where(V_pqr==self.V[i-1])[0][0], np.where(V_pqr==self.V[i])[0][0], np.where(V_pqr==self.V[i+1])[0][0]])
            tm_pqr[row, ind_pqr] = tp[i-1]

        if full:
            return tm
        else:
            return tm_pqr


    def nloglik(self, problems, data, pars):
        """For a single set of parameters, evaluate the
        log-likelihood of observed data set."""
        nllh = []
        nllh_choice = []

        excluded = ['fitting']
        nonspec = filter(lambda k: k.count('(')==0 and k not in excluded, pars)
        for k in nonspec:
            data.loc[:,k] = [pars[k] for _ in range(data.shape[0])]

        factors = []
        spec = filter(lambda k: k.count('(')>0, pars)
        for k in spec:
            sp = k.rstrip(')').split('(')
            p = sp[0]
            f, value = sp[1].split('=')
            factors.append(f)
            ss = data[data[f]==value]
            data.loc[data[f]==value,p] = [pars[k] for _ in range(ss.shape[0])]


        for i, probdata in data.groupby(['problem'] + factors):

            pars['max_T'] = probdata.samplesize.max() + 1
            pid = probdata['problem'].values[0]
            pars['probid'] = pid

            grppars = {}
            nonspec = filter(lambda k: k.count('(')==0, pars)
            for k in nonspec: grppars[k] = pars[k]

            spec = filter(lambda k: k.count('(')>0, pars)
            for k in spec:
                p = k.split('(')[0]
                grppars[p] = probdata[p].values[0]

            # run the model
            results = self.__call__(problems[pid], grppars)

            # decrement observed sample sizes to line up with
            # model (including any minimum sample size)
            minss = grppars.get('minsamplesize', 1)
            ss = np.array(probdata.samplesize.values, int) - minss

            choices = np.array(probdata.choice.values, int)

            nllh_choice.append(np.sum((np.log(pfix(results['p_resp'][choices])))))
            nllh.append(-1 * np.sum((np.log(pfix(results['p_resp'][choices])) + \
                        np.log(pfix(results['p_stop_cond'][ss, choices])))))

        return np.sum(nllh)


    def nloglik_opt(self, value, problems, data, pars):
        """Unpack arguments from fitting method (fit.py)
        and check they are valid before getting nloglik"""
        pars, fitting, verbose = unpack(value, pars)

        # check if the parameters are within allowed range
        if outside_bounds(value, fitting):
            return np.inf
        else:
            for v, p in zip(value, fitting.keys()):
                pars[p] = v
            nllh = self.nloglik(problems, data, pars)
            return nllh


class CHASEAlternateStoppingModel(CHASEModel):
    """
    CHASE with a fixed stopping rule. Can currently be
    either a 1) truncated, discretized Normal or
    2) geometric distribution.
    """

    def __init__(self, **kwargs):
        super(CHASEAlternateStoppingModel, self).__init__(**kwargs)

        stoprule = kwargs.get('stoprule', None)
        if stoprule is None:
            print 'No stopping rule specified!'
        elif stoprule is 'truncatednormal':
            self.stoprule = TruncatedNormal()
        elif stoprule is 'geometric':
            self.stoprule = Geometric()


    def set_statespace(self, pars):
        self.theta = int(pars.get('theta', 5))
        self.V = np.arange(-self.theta, self.theta+1, 1, dtype=int)
        self.m = len(self.V)
        return


    def __call__(self, options, pars):
        """Evaluate the model for a given set of parameters."""
        self.max_T = int(pars.get('max_T', 100))  # range of sample sizes
        self.set_statespace(pars) # threshold and state space
        self.dt = 1

        # evaluate the starting distribution
        Z = self.Z(self.m, {'theta': self.theta + 1, 'tau': pars.get('tau', .5)})

        # transition matrix with reflecting boundaries
        tm = self.transition_matrix_reflecting(options, pars)

        # if there was a minimum sample size, update the
        # initial distribution for that number of steps
        if 'minsamplesize' in pars:
            min_ss = pars.get('minsamplesize')
            if min_ss == 2:
                Z = Z * tm
            else:
                Z = Z * matrix_power(tm, min_ss - 1)
            assert np.isclose(Z.sum(), 1.)

        # evaluate evolution in states
        M = [Z * tm]
        for i in range(1, self.max_T):
            M.append( np.dot(M[-1], tm) )
        p_state_t = np.array(M).reshape((self.max_T, self.m))

        # choice probabilities across states
        R = np.zeros((self.m, 2))
        R[self.theta,:] = 0.5
        R[:self.theta,0] = 1
        R[(self.theta+1):,1] = 1
        p_LH = np.dot(p_state_t, R)

        # stopping distribution
        p_stop = self.stoprule.dist(pars)

        return {'states_t': p_state_t,
                'p_resp_t': p_LH,
                'p_stop': p_stop}


    def transition_matrix_reflecting(self, options, pars):
        """
        Transition matrix with reflecting boundaries.
        """
        d = self.drift(options, pars)
        tp = np.tile(self.transition_probs(d, pars), (self.m, 1))

        tm         = np.zeros((self.m, self.m), float)
        tm[0,:2]   = [tp[0,:2].sum(), tp[0,2]]
        tm[-1,-2:] = [tp[-1,-3], tp[-1,-2:].sum()]
        for i in range(1, self.m - 1):
            tm[i,i-1:i+2] = tp[i-1]
        return tm


    def nloglik(self, problems, data, pars):
        """For a single set of parameters, evaluate the
        log-likelihood of observed data set."""

        nllh = []

        # stopping rule distribution
        p_stop = self.stoprule.dist(pars)

        results = {}
        for pid in data.problem.unique():
            probdata = data[data.problem==pid]
            pars['max_T'] = probdata.samplesize.max() + 1
            pars['probid'] = pid
            results[pid] = self.__call__(problems[pid], pars)

        nllh = []
        for i, obs in data.iterrows():

            problem, samplesize, choice = obs['problem'], obs['samplesize'], obs['choice']

            # if minimum sample size, make correction to
            # observed sample size here
            samplesize = samplesize - 1
            if 'minsamplesize' in pars:
                samplesize = samplesize - (pars['minsamplesize'] - 1)

            pred = results[problem]
            p_choice = pred['p_resp_t'][samplesize][choice]
            nllh.append(-1 * (np.log(pfix(p_choice)) + np.log(pfix(p_stop[samplesize]))))

        return np.sum(nllh)



class CHASEOptionalStoppingSwitchingModel(CHASEModel):

    def __init__(self, **kwargs):
        super(CHASEOptionalStoppingSwitchingModel, self).__init__(**kwargs)
        self.drift = SwitchingCPTDriftModel(**kwargs)


    def __call__(self, options, pars):
        """Evaluate the model for a given set of parameters."""
        self.max_T = pars.get('max_T', 100)
        self.set_statespace(pars) # threshold and state space
        self.dt = pars.get('dt', 1) # size of time increment
        self.step = 1 # size of state space increment

        T = np.arange(1., self.max_T + 1) # range of sample sizes
        N = map(int, np.floor(T))

        # which option is sampled first? [0, 1, None]
        firstoption = pars.get('firstoption', None)


        self.vi_pqr = np.concatenate(([0, self.m - 1, self.m, 2*self.m - 1], np.arange(1, self.m - 1), np.arange(self.m+1, 2*self.m-1)))

        # evaluate the starting distribution
        Z = self.Z(self.m - 2, pars)

        # repeat Z for both options, and re-normalize
        if firstoption is None:
            Z = np.concatenate((Z, Z), axis=1)
        elif firstoption is 0:
            Z = np.concatenate((Z, np.matrix(np.zeros(self.m - 2))), axis=1)
        elif firstoption is 1:
            Z = np.concatenate((np.matrix(np.zeros(self.m - 2)), Z), axis=1)
        Z = Z / float(np.sum(Z))


        # the transition matrix
        tm_pqr = self.transition_matrix_PQR(options, pars)

        Q = tm_pqr[4:,4:]
        I = np.eye(2 * (self.m - 2))
        R = np.matrix(tm_pqr[4:,:4])
        IQ = np.matrix(linalg.inv(I - Q))

        S = np.zeros((self.max_T, 2*(self.m - 2), 2*(self.m - 2)))
        S[0] = np.eye(2*(self.m - 2))
        for i in range(1, self.max_T):
            S[i] = np.dot(S[i-1], Q)
        SR = np.tensordot(S, R, axes=1)
        states_t = np.dot(Z, S)

        # 1. overall response probabilities
        rp = np.array(Z * (IQ * R))[0]
        resp_prob = np.array([rp[0] + rp[2], rp[1] + rp[3]])

        # 2. response probability over time
        p_resp_t = np.dot(Z, SR)
        resp_prob_t = np.array([p_resp_t[:,0] + p_resp_t[:,2], p_resp_t[:,1] + p_resp_t[:,3]])

        # 3. predicted stopping points, conditional on choice
        p_tsteps = (Z * (IQ * IQ) * R)
        p_tsteps = np.array([np.sum([p_tsteps[0,0], p_tsteps[0,2]]), np.sum([p_tsteps[0,1], p_tsteps[0,3]])])
        p_tsteps = p_tsteps / resp_prob

        # 4. probability of stopping over
        p_stop_cond = p_resp_t
        p_stop_cond_sep = p_stop_cond
        p_stop_cond = np.array([np.sum([p_stop_cond[:,0], p_stop_cond[:,2]], 0), np.sum([p_stop_cond[:,1] + p_stop_cond[:,3]], 0)])
        p_stop_cond = np.array([p_stop_cond[0] / resp_prob[0], p_stop_cond[1] / resp_prob[1]])


        return {'states_t': states_t,
                'p_resp': resp_prob,
                'p_resp_sep': np.array(rp),
                'p_resp_t': resp_prob_t,
                'exp_samplesize': p_tsteps,
                'p_stop_cond': p_stop_cond.transpose()[0],
                'p_stop_cond_sep': p_stop_cond_sep}


    def transition_matrix_PQR(self, options, pars, full=False):
        p_switch = pars.get('p_switch', 0.5)

        # transition probabilities for each option
        tp = []
        for i in range(len(options)):
            d = self.drift(i, options, pars)
            tp.append(self.transition_probs(d, pars))

        PSTAY = {}
        PSWITCH = {}
        for i, opt in enumerate(options):

            other = [1, 0][i]
            PSTAY[i] = np.zeros((self.m, self.m), float)
            PSTAY[i][0,0] = 1.
            PSTAY[i][self.m-1, self.m-1] = 1.
            PSWITCH[i] = np.zeros((self.m, self.m), float)
            for row in range(1, self.m-1):
                PSTAY[i][row,row-1:row+2]   = (1 - p_switch) * tp[i]
                PSWITCH[i][row,row-1:row+2] =      p_switch  * tp[other]


        P = np.concatenate([np.concatenate([PSTAY[0], PSWITCH[0]], axis=1),
                            np.concatenate([PSWITCH[1], PSTAY[1]], axis=1)])


        tm_pqr = np.zeros((self.m * 2, self.m * 2), float)
        for i, vi in enumerate(self.vi_pqr):
            tm_pqr[i,:] = P[vi,:]

        tm = np.zeros((self.m * 2, self.m * 2), float)
        for i, vi in enumerate(self.vi_pqr):
            tm[:,i] = tm_pqr[:,vi]

        return tm



class CHASEAlternateStoppingSwitchingModel(CHASEAlternateStoppingModel):

    """This incorporates an alternate stopping rule"""

    def __init__(self, **kwargs):
        super(CHASEAlternateStoppingSwitchingModel, self).__init__(**kwargs)
        self.drift = SwitchingCPTDriftModel(**kwargs)


    def __call__(self, options, pars):
        """Evaluate the model for a given set of parameters."""
        self.max_T = int(pars.get('max_T', 100))    # range of sample sizes
        T = np.arange(1., self.max_T + 1)
        N = map(int, np.floor(T))
        self.set_statespace(pars)                   # threshold and state space

        # which option is sampled first? [0, 1, None]
        firstoption = pars.get('firstoption', None)

        # evaluate the starting distribution
        Z = self.Z(self.m, {'theta': self.theta + 1, 'tau': pars.get('tau', .5)})

        # repeat Z for both options, and re-normalize
        if firstoption is None:
            Z = np.concatenate((Z, Z), axis=1)
        elif firstoption is 0:
            Z = np.concatenate((Z, np.matrix(np.zeros(self.m - 2))), axis=1)
        elif firstoption is 1:
            Z = np.concatenate((np.matrix(np.zeros(self.m - 2)), Z), axis=1)
        Z = Z / float(np.sum(Z))


        # transition matrix
        tm = self.transition_matrix_reflecting(options, pars)

        # min-steps
        if 'minsamplesize' in pars:
            min_ss = pars.get('minsamplesize')
            if min_ss == 2:
                Z = Z * tm
            else:
                Z = Z * matrix_power(tm, min_ss - 1)
            assert np.isclose(Z.sum(), 1.)

        # evaluate evolution in states
        M = [Z * matrix_power(tm, 0)]
        for n in N[1:]:
            M.append( np.dot(M[-1], tm) )
        p_state_t = np.array(M).reshape((len(N), self.m * 2))

        p_0 = p_state_t[:,self.theta] + p_state_t[:,self.m + self.theta]
        p_L = p_state_t[:,:self.theta].sum(axis=1) + p_state_t[:,self.m:(self.m+self.theta)].sum(axis=1) + p_0 * 0.5
        p_H = p_state_t[:,(1+self.theta):self.m].sum(axis=1) + p_state_t[:,(self.m+self.theta+1):].sum(axis=1) + p_0 * 0.5
        p_LH = np.transpose((p_L, p_H))

        assert np.isclose(p_LH.sum(), p_LH.shape[0])

        return {'T': T,
                'states_t': p_state_t,
                'p_resp_t': p_LH}


    def transition_matrix_reflecting(self, options, pars, full=False):
        p_switch = pars.get('p_switch', 0.5)

        # transition probabilities for each option
        tp = []
        for i in range(len(options)):
            d = self.drift(i, options, pars)
            tp.append(self.transition_probs(d, pars))
        tp = np.array(tp)

        PSTAY = {}
        PSWITCH = {}
        for i, opt in enumerate(options):

            other = [1, 0][i]
            PSTAY[i]   = np.zeros((self.m, self.m), float)
            PSWITCH[i] = np.zeros((self.m, self.m), float)

            PSTAY[i][0,:2]     = (1 - p_switch) * np.array([tp[i,:2].sum(), tp[i,2]])
            PSTAY[i][-1,-2:]   = (1 - p_switch) * np.array([tp[i,0], tp[i,-2:].sum()])

            PSWITCH[i][0,:2]   = (p_switch) * np.array([tp[other,:2].sum(), tp[other,2]])
            PSWITCH[i][-1,-2:] = (p_switch) * np.array([tp[other,0], tp[other,-2:].sum()])

            for row in range(1, self.m-1):
                PSTAY[i][row,row-1:row+2]   = (1 - p_switch) * tp[i]
                PSWITCH[i][row,row-1:row+2] =      p_switch  * tp[other]

        tm = np.concatenate([np.concatenate([PSTAY[0], PSWITCH[0]], axis=1),
                             np.concatenate([PSWITCH[1], PSTAY[1]], axis=1)])

        return tm



if __name__ == '__main__':

    options = [[[  4.        ,   0.5       ],
                [ 12.        ,   0.26315789],
                [  0.        ,   0.23684211]],
               [[  1.  ,   0.41],
                [  0.  ,   0.28],
                [ 18.  ,   0.31]]]
    options = np.array(options)

    for drift in [.5, 'ev', 'cpt']:
        print drift
        m = CHASEModel(drift=drift)
        print m(options, {'prelec_gamma': 1.4})
