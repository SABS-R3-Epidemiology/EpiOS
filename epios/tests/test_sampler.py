import numpy as np
import pandas as pd
import unittest
from unittest import TestCase
from sampler import Sampler
import os
from pandas.testing import assert_frame_equal
from utils import person_allowed


class TestDataProcess(TestCase):

    def setUp(self) -> None:
        '''
        This function set up the testing environment
        Firstly use the DataProcess class to generate input for Sampler
        Secondly construct the Sampler class

        This function include some of the expected results

        '''
        self.path = './testing_sampler/'
        try:
            os.mkdir(self.path[2:-1])
        except FileExistsError:
            raise KeyError('Directory already exists, terminated not to overwrite anything!')
        self.data = pd.DataFrame({'ID': ['0.0.0.0', '0.0.0.1', '0.0.1.0',
                                         '0.1.0.0', '0.2.0.0', '1.0.0.0'],
                                  'age': [1, 81, 45, 33, 20, 60]})
        self.sampler = Sampler(data=self.data, data_store_path=self.path)

    def test_error(self):
        with self.assertRaises(ValueError):
            Sampler(data=1, data_path=1)
        with self.assertRaises(ValueError):
            Sampler()
        with self.assertRaises(ValueError):
            self.sampler.sample(len(self.sampler.data) + 1)

    def test__init__(self):
        try:
            self.init = Sampler(data_path=self.path + 'data.csv')
            assert_frame_equal(self.init.data, pd.read_csv(self.path + 'data.csv'))
        except AssertionError:
            self.fail('not load data as expected')

    def test_sample(self):
        np.random.seed(1)
        self.assertEqual(self.sampler.sample(len(self.sampler.data)), ['0.0.1.0', '0.0.0.1', '0.2.0.0',
                                                                       '0.0.0.0', '0.1.0.0', '1.0.0.0'])

    def tearDown(self) -> None:
        '''
        Clean up everything created

        '''
        if os.path.exists(self.path):
            if os.path.exists(self.path + 'pop_dist.json'):
                os.remove(self.path + 'pop_dist.json')
            if os.path.exists(self.path + 'microcells.csv'):
                os.remove(self.path + 'microcells.csv')
            if os.path.exists(self.path + 'data.csv'):
                os.remove(self.path + 'data.csv')
            os.rmdir(self.path)

    def test_person_allowed(self):
        sample = ["0.0.0.0", "0.0.0.1"]
        choice = "0.0.0.2"
        threshold = 3
        result = person_allowed(sample, choice, threshold)
        self.assertTrue(result)
        new_sample = ["0.0.0.0", "0.0.0.1", "0.0.0.3"]
        new_result = person_allowed(new_sample, choice, threshold)
        self.assertFalse(new_result)
        new_threshold = 2
        result = person_allowed(sample, choice, new_threshold)
        self.assertFalse(result)


if __name__ == '__main__':

    unittest.main()
