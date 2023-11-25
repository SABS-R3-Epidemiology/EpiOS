import numpy as np
import pandas as pd
import unittest
from unittest import TestCase
from epios import NonResponder
from epios import DataProcess
from numpy.testing import assert_array_equal
import os


class TestDataProcess(TestCase):

    def setUp(self) -> None:
        self.path = './testing_nonresponders/'
        try:
            os.mkdir(self.path[2:-1])
        except:
            raise KeyError('Directory already exists, terminated not to overwrite anything!')
        self.data = pd.DataFrame({'ID': ['0.0.0.0', '0.0.0.1', '0.0.1.0',
                                         '0.1.0.0', '0.2.0.0', '1.0.0.0'],
                                  'age': [1, 81, 45, 33, 20, 60]})
        self.processor = DataProcess(self.data)
        self.processor.pre_process(path=self.path)

        self.nonresponder = NonResponder(nonRespRate=[1] + [0] * (17 * 2 - 1),
                                         geoinfo_path=self.path + 'microcells.csv',
                                         ageinfo_path=self.path + 'pop_dist.json',
                                         data_path=self.path + 'data.csv')

    def test__init__(self):
        with self.assertRaises(ValueError):
            NonResponder(nonRespRate=[1], geoinfo_path=self.path + 'microcells.csv',
                         ageinfo_path=self.path + 'pop_dist.json',
                         data_path=self.path + 'data.csv')

    def test_additional_sample(self):
        expected_res = np.zeros((2, 17))
        expected_res[0, 0] = 1
        try:
            assert_array_equal(np.array(self.nonresponder.additional_sample(1, 1)), expected_res)
        except:
            self.fail('additional samples not generated as expected')
        try:
            assert_array_equal(np.array(self.nonresponder.additional_sample(1, 0, 1)), expected_res)
        except:
            self.fail('additional samples not generated as expected')

    def tearDown(self) -> None:
        if os.path.exists(self.path):
            if os.path.exists(self.path + 'pop_dist.json'):
                os.remove(self.path + 'pop_dist.json')
            if os.path.exists(self.path + 'microcells.csv'):
                os.remove(self.path + 'microcells.csv')
            if os.path.exists(self.path + 'data.csv'):
                os.remove(self.path + 'data.csv')
            os.rmdir(self.path)


if __name__ == '__main__':

    unittest.main()
