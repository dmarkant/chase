from copy import deepcopy
import numpy as np
from scipy import linalg
from numpy.linalg import matrix_power
from drift import DriftModel, CPTDriftModel
from stopping import TruncatedNormal
from initial_distribution import *
from utils import pfix


class CHASEModel(object):
    """CHASEModel implements the basic sequential optional stopping
    sampling model with a specified function for evaluating
    the drift rate."""

    def __init__(self, **kwargs):
        """Instantiate the optional stopping model.

        :Arguments:

            * data: pandas.DataFrame
                data containing 'n_samples_A', 'n_samples_B', and 'choice' columns,
                as well as any further columns to specify grouping variable

            * verbose:

        """
        self.data = kwargs.get('data', None)
        self.verbose = kwargs.get('verbose', False) # replace with logging

        # set function for drift rate
        drift = kwargs.get('drift', 'ev')
        if isinstance(drift, float) or isinstance(drift, int):
            self.drift = lambda *args, **kwargs: drift
        if drift is 'ev':
            self.drift = DriftModel()
        elif drift is 'cpt':
            self.drift = CPTDriftModel()

        # set function for initial distribution
        startdist = kwargs.get('startdist', 'uniform')
        if 'numpy' in str(type(startdist)):
            self.Z = lambda *args, **kwargs: startdist
        elif startdist is 'uniform':
            self.Z = uniform_initial_distribution
        elif startdist is 'softmax':
            self.Z = softmax_initial_distribution
        elif startdist is 'indifferent':
            self.Z = indifferent_initial_distribution


    def __call__(self, options, pars):
        """Evaluate the model for a given set of parameters."""

        self.max_T = pars.get('max_T', 100)   # range of timesteps to evaluate over
        T = np.arange(1., self.max_T + 1)
        N = map(int, np.floor(T))


        # threshold and state space
        self.theta = np.float(np.round(pars.get('theta', 5)))     # boundaries
        self.V = np.round(np.arange(-self.theta, self.theta+(1/2.), 1), 4)
        self.vi = range(len(self.V))
        self.m = len(self.V)

        vi_pqr = []
        start = np.array([[0, self.m - 1], range(1, self.m - 1)])
        for outer in start:
            for inner in outer:
                vi_pqr.append(inner)
        self.vi_pqr = np.array(vi_pqr)
        self.V_pqr = self.V[vi_pqr] # sort state space


        # evaluate the starting distribution
        Z = self.Z(self.m - 2, pars)


        # transition matrix
        tm_pqr = self.transition_matrix_PQR(options, pars)
        Q = tm_pqr[2:,2:]
        I = np.eye(self.m - 2)
        R = np.matrix(tm_pqr[2:,:2])
        IQ = np.matrix(linalg.inv(I - Q))


        # min-steps
        if 'min_steps' in pars:
            Z = Z * matrix_power(Q, pars.get('min_steps') - 1)
            Z = np.matrix(Z / np.sum(Z))

        S = [matrix_power(Q, 0)]
        for n in N[1:]:
            S.append(np.dot(S[-1], Q))
        S = np.array(S)
        SR = np.array([np.dot(s, R) for s in S])
        states_t = np.dot(Z, S)

        # overall response probabilities
        # (see Diederich and Busemeyer, 2003, eqn. 2)
        p_resp = Z * (IQ * R)

        # response probability over timesteps
        p_resp_t = np.array([np.dot(Z, sr) for sr in SR]).reshape((len(N), 2))

        # probability of stopping over time
        # (see Diederich and Busemeyer, 2003, eqn. 18)
        p_stop_cond = np.array([np.dot(Z, sr)/p_resp for sr in SR]).reshape((len(N), 2))

        # expected number of timesteps, conditional on choice
        # (see Diederich and Busemeyer, 2003, eqn. 3)
        exp_samplesize = (Z*(IQ*IQ)*R)/p_resp


        return {'T': T,
                'states_t': states_t,
                'p_resp': np.array(p_resp)[0],
                'p_resp_t': p_resp_t,
                'p_stop_cond': p_stop_cond,
                'exp_samplesize': exp_samplesize,
                }


    def transition_probs(self, options, pars, state=None):
        d = self.drift(options, pars, state=None)
        p_stay = np.min([.9999, pars.get('p_stay', 1./3.)])
        p_down = ((1 - p_stay)/2.) * (1 - d)
        p_up = ((1 - p_stay)/2.) * (1 + d)
        assert np.round(np.sum([p_down, p_stay, p_up]), 5)==1.
        return [p_down, p_stay, p_up]


    def transition_matrix_PQR(self, options, pars):
        gamma = pars.get('gamma', 0.)

        tm_pqr = np.zeros((self.m, self.m), float)
        tm_pqr[0,0] = 1.
        tm_pqr[1,1] = 1.

        # if there is state-dependent weighting, compute
        # transition probabilities for each state. Otherwise,
        # use same transition probabilities for everything
        if gamma == 0.:
            tp = np.tile(self.transition_probs(options, pars), (self.m - 2, 1))
        else:
            tp = np.array([self.transition_probs(options, pars, state=i) for i in range(1, self.m - 1)])

        tm         = np.zeros((self.m, self.m), float)
        tm[0,0] = 1.
        tm[-1,-1] = 1.

        # construct PQR row by row
        V_pqr = deepcopy(self.V_pqr)
        for i in range(1, self.m - 1):
            tm[i,i-1:i+2] = tp[i-1]

            row = np.where(V_pqr==self.V[i])[0][0]
            ind_pqr = np.array([np.where(V_pqr==self.V[i-1])[0][0], np.where(V_pqr==self.V[i])[0][0], np.where(V_pqr==self.V[i+1])[0][0]])
            tm_pqr[row, ind_pqr] = tp[i-1]

        return tm_pqr


    def nloglik(self, problems, data, pars):
        """For a single set of parameters, evaluate the
        log-likelihood of observed data set."""

        results = {pid: self.__call__(problems[pid], pars) \
                   for pid in data['problem'].unique()}

        nllh = []
        for i, obs in data.iterrows():

            problem, samplesize, choice = obs['problem'], obs['samplesize'], obs['choice']
            pred = results[problem]

            # if there is a minimum sample size,
            # make correction to observed sample
            # size here

            nllh.append(-1 * (np.log(pfix(pred['p_resp'][choice])) + \
                        np.log(pfix(pred['p_stop_cond'][samplesize - 1, choice]))))

        return np.sum(nllh)



class CHASEAlternateStoppingModel(CHASEModel):

    """This incorporates an alternate stopping rule"""

    def __init__(self, **kwargs):
        super(CHASEAlternateStoppingModel, self).__init__()

        stoprule = kwargs.get('stoprule', None)
        if stoprule is None:
            print 'No stopping rule specified!'
        elif stoprule is 'truncatednormal':
            self.stoprule = TruncatedNormal()


    def __call__(self, options, pars):
        """Evaluate the model for a given set of parameters."""

        self.max_T = pars.get('max_T', 100)   # range of timesteps to evaluate over
        T = np.arange(1., self.max_T + 1)
        N = map(int, np.floor(T))


        # threshold and state space
        self.theta = np.float(np.round(pars.get('theta', 5)))     # boundaries
        self.V = np.round(np.arange(-self.theta, self.theta+(1/2.), 1), 4)
        self.vi = range(len(self.V))
        self.m = len(self.V)

        vi_pqr = []
        start = np.array([[0, self.m - 1], range(1, self.m - 1)])
        for outer in start:
            for inner in outer:
                vi_pqr.append(inner)
        self.vi_pqr = np.array(vi_pqr)
        self.V_pqr = self.V[vi_pqr] # sort state space


        # evaluate the starting distribution
        Z = self.Z(self.m - 2, pars)

        # transition matrix
        tm = self.transition_matrix_reflecting(pars)

        # min-steps
        if 'min_steps' in pars:
            Z = Z * matrix_power(tm, pars.get('min_steps') - 1)
            assert np.round(np.sum(Z), 5)==1.

        M = [Z * matrix_power(tm, 0)]
        for n in N[1:]:
            M.append( np.dot(M[-1], tm) )
        p_state_t = np.array(M).reshape((len(N), self.m))

        p_0 = p_state_t[:,self.theta]
        p_L = p_state_t[:,:self.theta].sum(axis=1) + p_0 * 0.5
        p_H = p_state_t[:,(1+self.theta):].sum(axis=1) + p_0 * 0.5
        p_LH = np.transpose((p_L, p_H))

        return {'T': T,
                'states_t': p_state_t,
                'p_resp_t': p_LH}


    def transition_matrix_reflecting(self, pars):
        """
        Transition matrix with reflecting boundaries.

        V -- discrete state space
        dv -- step size
        gamma -- state-dependent weight
        """
        p_stay = pars.get('p_stay', .5)
        gamma = pars.get('gamma', 0.)

        # if there is state-dependent weighting, compute
        # transition probabilities for each state. Otherwise,
        # use same transition probabilities for everything
        if gamma == 0.:
            tp = np.tile(transition_probs(pars, self.tau), (self.m, 1))
        else:
            tp = np.array([transition_probs(pars, self.tau, state=i) for i in range(self.m)])

        tm         = np.zeros((self.m, self.m), float)
        tm[0,:2]   = [tp[0,:2].sum(), tp[0,2]]
        tm[-1,-2:] = [tp[-1,-3], tp[-1,-2:].sum()]
        for i in range(1, self.m - 1):
            tm[i,i-1:i+2] = tp[i-1]
        return tm


    def nloglik(self, problems, data, pars):
        """For a single set of parameters, evaluate the
        log-likelihood of observed data set."""

        # random walk prediction
        results = {pid: self.__call__(problems[pid], pars) \
                   for pid in data['problem'].unique()}

        # stopping rule distribution
        p_stop = self.stoprule.nloglik(pars)

        nllh = []
        for i, obs in data.iterrows():

            problem, samplesize, choice = obs['problem'], obs['samplesize'], obs['choice']

            # if minimum sample size, make correction to
            # observed sample size here

            pred = results[problem]
            p_choice = pred['p_resp_t'][samplesize][choice]
            nllh.append(-1 * (np.log(pfix(p_choice)) + np.log(pfix(p_stop[samplesize]))))

        return np.sum(nllh)




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
