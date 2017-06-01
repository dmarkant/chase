import unittest
import initial_distribution as initdist

N_1 = 11
N_2 = 89

pars_1 = {'tau': .001, 'theta': 5}
pars_2 = {'tau': .081, 'theta': 44}

class TestINITDIST(unittest.TestCase):

    def setUp(self):
        pass

    def test_sum_probs_iid_1(self):
        self.assertEqual(initdist.indifferent_initial_distribution(N_1).sum(),
        1.)

    def test_sum_probs_iid_2(self):
        self.assertEqual(initdist.indifferent_initial_distribution(N_2).sum(),
        1.)

    def test_sum_probs_uid_1(self):
        self.assertEqual(initdist.uniform_initial_distribution(N_1).sum(),
        1.)

    def test_sum_probs_uid_2(self):
        self.assertEqual(initdist.uniform_initial_distribution(N_2).sum(),
        1.)

    def test_sum_probs_sid_1(self):
        self.assertEqual(round(initdist.softmax_initial_distribution(N_1, pars_1).sum(),
        10), 1.)

    def test_sum_probs_sid_2(self):
        self.assertEqual(round(initdist.softmax_initial_distribution(N_2, pars_2).sum(),
        10), 1.)

    def test_sum_probs_lid_1(self):
        self.assertEqual(round(initdist.laplace_initial_distribution(N_1, pars_1).sum(),
        10), 1.)

    def test_sum_probs_lid_2(self):
        self.assertEqual(round(initdist.laplace_initial_distribution(N_2, pars_2).sum(),
        10), 1.)


if __name__ == '__main__':
    unittest.main()
