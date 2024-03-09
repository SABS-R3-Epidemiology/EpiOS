from unittest import TestCase
from numpy.random import rand
from numpy import array
from re_scaler import ReScaler


class TestRS(TestCase):

    def test_call(self):
        x = rand()
        with self.assertRaises(Exception):
            ReScaler(false_positive=x, false_negative=1 - x)
        self.assertEqual(ReScaler()((x, 1 - x, 1)), x)
        self.assertEqual(ReScaler(false_positive=1, false_negative=1)((x, 1 - x, 1)), 1 - x)

    def test_smooth(self):
        x = [([1], [7], [1]), ([1, 2], [7, 6], [1, 1]), ([1, 2, 3], [7, 6, 5], [1, 1, 1])]
        with self.assertRaises(Exception):
            ReScaler(smoothing=lambda x: 1)(x)
        result = ReScaler(smoothing=lambda x: 1)(x, times=array([0, 1, 2]))
        self.assertEqual(result[0], 0.125)
        self.assertEqual(result[1], 0.25)
        self.assertEqual(result[2], 0.375)
