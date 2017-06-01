import unittest
import cpt
import numpy as np

params1 = {'pow_gain': 2,
          'pow_loss': 2,
          'w_loss': 1}

params2 = {'pow_gain': 2,
            'w_loss': 1}

outcome1 = np.array([1,2,3])
outcome2 = np.array([10,11,22,44])

class TestCPT(unittest.TestCase):

    def setUp(self):
        pass

    def test_value_fnc_1(self):
        self.assertEqual((cpt.value_fnc(outcome1, params1) +
                         cpt.value_fnc(-1 * outcome1, params1)).sum(), 0)

    def test_value_fnc_2(self):
        self.assertEqual((cpt.value_fnc(-1 * outcome1, params1) +
                         cpt.value_fnc(outcome1, params1)).sum(), 0)

    def test_value_fnc_3(self):
        self.assertEqual((cpt.value_fnc(outcome2, params2) +
                         cpt.value_fnc(-1 * outcome2, params2)).sum(), 0)

    def test_value_fnc_4(self):
        self.assertEqual((cpt.value_fnc(-1 * outcome2, params2) +
                         cpt.value_fnc(outcome2, params2)).sum(), 0)


if __name__ == '__main__':
    unittest.main()
