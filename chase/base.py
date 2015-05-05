import numpy as np
from drift import DriftModel, CPTDriftModel

class CHASEModel(object):
    """CHASEModel implements the basic sequential optional stopping
    sampling model with a specified function for evaluating
    the drift rate."""

    def __init__(self, data=None, pars={}, **kwargs):
        """Instantiate the optional stopping model.

        :Arguments:

            * data: pandas.DataFrame
                data containing 'n_samples_A', 'n_samples_B', and 'choice' columns,
                as well as any further columns to specify grouping variable

            * pars:

        """
        self.verbose = kwargs.get('verbose', False)
        self.pars = pars
        self.data = data

        # fixed parameters
        self.theta = np.float(np.round(pars.get('theta', 5)))     # boundaries
        self.V = np.round(np.arange(-self.theta, self.theta+(1/2.), 1), 4)
        self.vi = range(len(self.V))
        self.m = len(self.V)

        drift = kwargs.get('drift', 'ev')
        if isinstance(drift, float) or isinstance(drift, int):
            self.drift = lambda *args, **kwargs: drift
        if drift is 'ev':
            self.drift = DriftModel()
        elif drift is 'cpt':
            self.drift = CPTDriftModel()


        # state space
        vi_pqr = []
        start = np.array([[0, self.m - 1], range(1, self.m - 1)])
        for outer in start:
            for inner in outer:
                vi_pqr.append(inner)
        self.vi_pqr = np.array(vi_pqr)
        self.V_pqr = self.V[vi_pqr] # sort state space


    def __call__(self, pars):
        """Evaluate the model for a given set of parameters."""
        pass


    def transition_probs(self, pars):
        p_stay = np.min([.9999, pars.get('p_stay', 1./3.)])
        p_down = ((1 - p_stay)/2.) * (1 - d)
        p_up = ((1 - p_stay)/2.) * (1 + d)
        assert np.round(np.sum([p_down, p_stay, p_up]), 5)==1.
        return [p_down, p_stay, p_up]


    def transition_matrix(self, pars):
        """Transition matrix."""
        raise NotImplementedError


    def transition_matrix_PQR(self, pars):
        raise NotImplementedError





if __name__ == '__main__':

    m = CHASEModel(drift=.5)

    options = [[[  4.        ,   0.5       ],
                [ 12.        ,   0.26315789],
                [  0.        ,   0.23684211]],
               [[  1.  ,   0.41],
                [  0.  ,   0.28],
                [ 18.  ,   0.31]]]


