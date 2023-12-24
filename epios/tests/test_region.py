import numpy as np
import pandas as pd
import unittest
from unittest import TestCase
from sampler_region import SamplerRegion
import os
from numpy.testing import assert_array_equal
# from pandas.testing import assert_frame_equal


class TestDataProcess(TestCase):

    def setUp(self) -> None:
        '''
        This function set up the testing environment
        Firstly use the DataProcess class to generate input for Sampler
        Secondly construct the Sampler class

        This function include some of the expected results

        '''
        self.path = './testing_region/'
        try:
            os.mkdir(self.path[2:-1])
        except FileExistsError:
            raise FileExistsError('Directory already exists, terminated not to overwrite anything!')
        self.data = pd.DataFrame({'ID': ['0.0.0.0', '0.0.0.1', '0.0.1.0',
                                         '0.1.0.0', '0.2.0.0', '1.0.0.0']})
        self.sampler = SamplerRegion(data=self.data, data_store_path=self.path)

        self.expected_region_dist = [5 / 6, 1 / 6]

    def test_get_region_dist(self):
        self.assertEqual(self.sampler.get_region_dist(), self.expected_region_dist)

    def test_bool_exceed(self):
        self.assertEqual(self.sampler.bool_exceed(1, 2), False)
        self.assertEqual(self.sampler.bool_exceed(0, 2), True)

    def test_multinomial_draw(self):
        np.random.seed(1)
        region_dist = self.sampler.get_region_dist()
        with self.assertRaises(ValueError):
            self.sampler.multinomial_draw(len(self.sampler.data) + 1, region_dist)

        with self.assertRaises(KeyError):
            self.sampler.multinomial_draw(len(self.sampler.data), [1] + [0] * (len(region_dist) - 1))

        res, cap = self.sampler.multinomial_draw(len(self.sampler.data), region_dist)
        try:
            assert_array_equal(res, np.array(cap))
        except AssertionError:
            self.fail('not draw as expected')

    def test_sample1(self):
        np.random.seed(1)
        self.assertEqual(self.sampler.sample(len(self.sampler.data)), ['0.0.1.0', '0.0.0.0', '0.0.0.1',
                                                                       '0.1.0.0', '0.2.0.0', '1.0.0.0'])
        np.random.seed(1)
        additional_sample = [0] * len(self.sampler.get_region_dist())
        additional_sample[0] = 1
        self.assertEqual(self.sampler.sample(len(self.sampler.data), additional_sample),
                         ['0.0.1.0', '0.0.0.0', '0.0.0.1', '0.1.0.0', '0.2.0.0', '1.0.0.0'])

    def test_sample2(self):
        self.data2 = pd.DataFrame({'ID': ['0.0.0.0', '0.0.0.1', '0.0.0.2',
                                          '0.0.0.3', '0.0.0.4', '0.0.0.5']})
        self.sampler2 = SamplerRegion(data=self.data2, data_store_path=self.path)
        np.random.seed(1)
        self.assertEqual(self.sampler2.sample(4, household_criterion=True, household_threshold=4),
                         ['0.0.0.0', '0.0.0.1', '0.0.0.3', '0.0.0.2'])
        with self.assertRaises(ValueError):
            self.sampler2.sample(len(self.sampler2.data), household_criterion=True, household_threshold=1)

    def test_additional_nonresponder(self):
        self.data3 = pd.DataFrame({'ID': ['0.0.0.0', '0.0.0.1', '0.0.1.0', '0.1.0.0',
                                          '0.2.0.0', '1.0.0.0'],
                                   'cell': [0, 0, 0, 0, 0, 1],
                                   'microcell': [0, 0, 0, 1, 2, 0],
                                   'household': [0, 0, 1, 0, 0, 0]})
        self.sampler3 = SamplerRegion(data=self.data3, data_store_path=self.path)
        expected_res = [5, 0]
        try:
            assert_array_equal(np.array(self.sampler3.additional_nonresponder(['0.0.0.0'], 1, 1)), expected_res)
        except AssertionError:
            self.fail('additional samples not generated as expected')
        try:
            assert_array_equal(np.array(self.sampler3.additional_nonresponder(['0.0.0.0'], 1, 0, 1)), expected_res)
        except AssertionError:
            self.fail('additional samples not generated as expected')

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


if __name__ == '__main__':

    unittest.main()
