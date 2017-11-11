import unittest
import sys
sys.path.append('../')
import cpt

params1 = {'pow_gain': 2,
          'pow_loss': 2,
          'w_loss': 1}

params2 = {'pow_gain': 2,
            'w_loss': 2}

outcome1 = 2

class TestCPT(unittest.TestCase):

    def setUp(self):
        pass

    def test_value_fnc_1(self):
        self.assertEqual(cpt.value_fnc(outcome1, params1) +
                         cpt.value_fnc(-1 * outcome1, params1), 0)
    def test_value_fnc_2(self):
        self.assertEqual(cpt.value_fnc(-1 * outcome1, params1) +
                         cpt.value_fnc(outcome1, params1), 0)
    def test_value_fnc_3(self):
        self.assertEqual(cpt.value_fnc(-1 * outcome1, params2) /
                         cpt.value_fnc(outcome1, params2), -params2.get('w_loss'))

if __name__ == '__main__':
    unittest.main()
