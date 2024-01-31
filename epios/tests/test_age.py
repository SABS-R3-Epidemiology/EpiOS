import numpy as np
import pandas as pd
import unittest
from unittest import TestCase
from sampler_age import SamplerAge
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
        self.path = './testing_age/'
        try:
            os.mkdir(self.path[2:-1])
        except FileExistsError:
            raise FileExistsError('Directory already exists, terminated not to overwrite anything!')
        self.data = pd.DataFrame({'id': ['0.0.0.0', '0.0.0.1', '0.0.1.0',
                                         '0.1.0.0', '0.2.0.0', '1.0.0.0'],
                                  'age': [1, 81, 45, 33, 20, 60]})
        self.sampler = SamplerAge(data=self.data, data_store_path=self.path)
        self.expected_age_dist = [1 / 6, 0.0, 0.0, 0.0, 1 / 6, 0.0, 1 / 6, 0.0,
                                  0.0, 1 / 6, 0.0, 0.0, 1 / 6, 0.0, 0.0, 0.0, 1 / 6]

    def test_get_age_dist(self):
        self.assertEqual(self.sampler.get_age_dist(), self.expected_age_dist)

    def test_bool_exceed(self):
        self.assertEqual(self.sampler.bool_exceed(1, 2), False)
        self.assertEqual(self.sampler.bool_exceed(0, 2), True)

    def test_multinomial_draw(self):
        np.random.seed(1)
        age_dist = self.sampler.get_age_dist()
        with self.assertRaises(ValueError):
            self.sampler.multinomial_draw(len(self.sampler.data) + 1, age_dist)

        with self.assertRaises(KeyError):
            self.sampler.multinomial_draw(len(self.sampler.data), [1] + [0] * (len(age_dist) - 1))

        res = self.sampler.multinomial_draw(len(self.sampler.data), age_dist)
        try:
            assert_array_equal(res, np.array([1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]))
        except AssertionError:
            self.fail('not draw as expected')

    def test_sample1(self):
        np.random.seed(1)
        self.assertEqual(self.sampler.sample(len(self.sampler.data)), ['0.0.0.0', '0.2.0.0', '0.1.0.0',
                                                                       '0.0.1.0', '1.0.0.0', '0.0.0.1'])

    def test_sample2(self):
        self.data1 = pd.DataFrame({'id': ['0.0.0.0', '0.0.0.1', '0.0.1.0',
                                          '0.1.0.0', '0.2.0.0', '1.0.0.0'],
                                   'age': [1, 2, 45, 33, 20, 60]})
        self.sampler1 = SamplerAge(data=self.data1, data_store_path=self.path)
        np.random.seed(1)
        self.assertEqual(self.sampler1.sample(len(self.sampler1.data)), ['0.0.0.1', '0.0.0.0', '0.2.0.0',
                                                                         '0.1.0.0', '0.0.1.0', '1.0.0.0'])

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
