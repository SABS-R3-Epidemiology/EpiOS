import numpy as np
import pandas as pd
import unittest
from unittest import TestCase
from sampler import Sampler
import os
from pandas.testing import assert_frame_equal


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
            Sampler()
        with self.assertRaises(ValueError):
            self.sampler.sample(len(self.sampler.data) + 1)

    def test_sample(self):
        np.random.seed(1)
        self.assertEqual(self.sampler.sample(len(self.sampler.data)), ['0.0.1.0', '0.0.0.1', '0.2.0.0',
                                                                       '0.0.0.0', '0.1.0.0', '1.0.0.0'])

    def test__init__(self):
        self.sampler1 = Sampler(data_store_path=self.path, pre_process=False)
        try:
            assert_frame_equal(self.sampler1.data, self.data)
        except AssertionError:
            self.fail('Initiation in the Base mode is unexpected')

    def test_person_allowed(self):
        sample = ["0.0.0.0", "0.0.0.1"]
        choice = "0.0.0.2"
        threshold = 3
        result = self.sampler.person_allowed(sample, choice, threshold)
        self.assertTrue(result)
        new_sample = ["0.0.0.0", "0.0.0.1", "0.0.0.3"]
        new_result = self.sampler.person_allowed(new_sample, choice, threshold)
        self.assertFalse(new_result)
        new_threshold = 2
        result = self.sampler.person_allowed(sample, choice, new_threshold)
        self.assertFalse(result)

    def tearDown(self) -> None:
        '''
        Clean up everything created

        '''
        if os.path.exists(self.path):
            for file in ['pop_dist.json', 'microcells.csv', 'data.csv']:
                if os.path.exists(self.path + file):
                    os.remove(self.path + file)
            os.rmdir(self.path)


if __name__ == '__main__':

    unittest.main()
