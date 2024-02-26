from unittest import TestCase
from numpy.random import rand
from numpy import array
from epios.re_scaler import ReScaler


class TestRS(TestCase):

    def test_call(self):
        x = rand()
        try:
            ReScaler(false_positive=x, false_negative=1 - x)
            raise Exception('shall not work')
        except:
            self.assertEqual(ReScaler()(x), x)
            self.assertEqual(ReScaler(false_positive=1, false_negative=1)(x), 1 - x)

    def test_smooth(self):
        x = [[1.0], [2.0, 2.0], [3.0, 3.0, 3.0]]
        try:
            ReScaler(smoothing=lambda x: 1)(x)
            raise Exception('shall not work')
        except:
            self.assertTrue((ReScaler(smoothing=lambda x: 1)(x, times=array([0.0, 1.0, 2.0]), tested=[[1],[1,1],[1,1,1]]) == array([1.0,2.0,3.0])).all())
