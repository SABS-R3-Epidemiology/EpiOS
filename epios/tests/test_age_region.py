import numpy as np
import pandas as pd
import unittest
from unittest import TestCase
from epios import DataProcess
from epios import Sampler
import os
from numpy.testing import assert_array_equal
# from pandas.testing import assert_frame_equal


class TestDataProcess(TestCase):

    def setUp(self) -> None:
        self.path = './testing_ageregion/'
        try:
            os.mkdir(self.path[2:-1])
        except:
            raise KeyError('Directory already exists, terminated not to overwrite anything!')
        self.data = pd.DataFrame({'ID': ['0.0.0.0', '0.0.0.1', '0.0.1.0', '0.1.0.0', '0.2.0.0', '1.0.0.0'], 'age': [1, 81, 45, 33, 20, 60]})
        self.processor = DataProcess(self.data)
        self.processor.pre_process(path=self.path)

        self.sampler = Sampler(geoinfo_path=self.path + 'microcells.csv', ageinfo_path=self.path + 'pop_dist.json', data_path=self.path + 'data.csv')

        self.expected_age_dist = [1 / 6, 0.0, 0.0, 0.0, 1 / 6, 0.0, 1 / 6, 0.0, 0.0, 1 / 6, 0.0, 0.0, 1 / 6, 0.0, 0.0, 0.0, 1 / 6]
        self.expected_region_dist = [5 / 6, 1 / 6]
        # self.expected_json = [1 / 6, 0.0, 0.0, 0.0, 1 / 6, 0.0, 1 / 6, 0.0, 0.0, 1 / 6, 0.0, 0.0, 1 / 6, 0.0, 0.0, 0.0, 1 / 6]
        # self.expected_df_microcell = pd.DataFrame({'cell': [0, 0, 0, 0, 1], 'microcell': [0, 0, 1, 2, 0], 'household': [0, 1, 0, 0, 0], 'Susceptible': [2, 1, 1, 1, 1]})
        # self.expected_df_population = pd.DataFrame({'ID': ['0.0.0.0', '0.0.0.1', '0.0.1.0', '0.1.0.0', '0.2.0.0', '1.0.0.0'], 'age': [1, 81, 45, 33, 20, 60], 'cell': [0, 0, 0, 0, 0, 1], 'microcell': [0, 0, 0, 1, 2, 0], 'household': [0, 0, 1, 0, 0, 0]})

    def test_get_age_dist(self):
        self.assertEqual(self.sampler.get_age_dist(), self.expected_age_dist)

    def test_get_region_dist(self):
        self.assertEqual(self.sampler.get_region_dist(), self.expected_region_dist)

    def test_bool_exceed(self):
        self.assertEqual(self.sampler.bool_exceed(1, 0, 0, 2, 2, 2), False)
        self.assertEqual(self.sampler.bool_exceed(0, 1, 0, 2, 2, 2), False)
        self.assertEqual(self.sampler.bool_exceed(0, 0, 1, 2, 2, 2), False)
        self.assertEqual(self.sampler.bool_exceed(0, 0, 0, 2, 2, 2), True)

    def test_multinomial_draw(self):
        np.random.seed(1)
        age_dist = self.sampler.get_age_dist()
        region_dist = self.sampler.get_region_dist()
        ar_dist = np.array(age_dist) * np.array(region_dist).reshape((-1, 1))
        ar_dist = ar_dist.reshape((1, -1))[0]
        with self.assertRaises(ValueError):
            self.sampler.multinomial_draw(len(self.sampler.data) + 1, ar_dist)

        with self.assertRaises(KeyError):
            self.sampler.multinomial_draw(len(self.sampler.data), [1] + [0] * (len(age_dist) * len(region_dist) - 1))

        res, cap = self.sampler.multinomial_draw(len(self.sampler.data), ar_dist)
        try:
            assert_array_equal(res, np.array(cap).reshape((1, -1))[0])
        except:
            self.fail('not draw as expected')

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
