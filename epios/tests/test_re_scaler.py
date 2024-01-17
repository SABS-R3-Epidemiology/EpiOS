from math import nan
import pandas as pd
import unittest
from unittest import TestCase
from numpy.random import rand
from epios.re_scaler import ReScaler


class TestRS(TestCase):

    def test__call__(self):
        x = rand()
        self.assertEqual(ReScaler()(x),x)
        self.assertEqual(ReScaler(false_positive=1, false_negative=1)(x),1-x)
        try:
            ReScaler(false_positive=x, false_negative=1-x)
            raise Exception('shall not work')
        except: pass