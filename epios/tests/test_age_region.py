import numpy as np
import pandas as pd
import unittest
from unittest import TestCase
from sampler_age_region import SamplerAgeRegion
import os
from numpy.testing import assert_array_equal
from unittest.mock import patch
# from pandas.testing import assert_frame_equal


class TestDataProcess(TestCase):

    def setUp(self) -> None:
        '''
        This function set up the testing environment
        Firstly use the DataProcess class to generate input for Sampler
        Secondly construct the Sampler class

        This function include some of the expected results

        '''
        self.path = './testing_ageregion/'
        try:
            os.mkdir(self.path[2:-1])
        except FileExistsError:
            raise FileExistsError('Directory already exists, terminated not to overwrite anything!')
        self.data = pd.DataFrame({'id': ['0.0.0.0', '0.0.0.1', '0.0.1.0',
                                         '0.1.0.0', '0.2.0.0', '1.0.0.0'],
                                  'age': [1, 81, 45, 33, 20, 60]})
        self.sampler = SamplerAgeRegion(data=self.data, data_store_path=self.path)

        self.expected_age_dist = [1 / 6, 0.0, 0.0, 0.0, 1 / 6, 0.0, 1 / 6, 0.0,
                                  0.0, 1 / 6, 0.0, 0.0, 1 / 6, 0.0, 0.0, 0.0, 1 / 6]
        self.expected_region_dist = [5 / 6, 1 / 6]

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
        except AssertionError:
            self.fail('not draw as expected')

    def test_sample1(self):
        np.random.seed(1)
        self.assertEqual(self.sampler.sample(len(self.sampler.data)), ['0.0.0.0', '0.2.0.0', '0.1.0.0',
                                                                       '0.0.1.0', '0.0.0.1', '1.0.0.0'])
        np.random.seed(1)
        additional_sample = np.zeros((len(self.sampler.get_region_dist()), len(self.sampler.get_age_dist())))
        additional_sample[0, 0] = 1
        self.assertEqual(self.sampler.sample(len(self.sampler.data), additional_sample),
                         ['0.0.0.0', '0.2.0.0', '0.1.0.0', '0.0.1.0', '0.0.0.1', '1.0.0.0'])

    def test_sample2(self):
        self.data1 = pd.DataFrame({'id': ['0.0.0.0', '0.0.0.1', '0.0.1.0',
                                          '0.1.0.0', '0.2.0.0', '1.0.0.0'],
                                  'age': [1, 2, 45, 33, 20, 60]})
        self.sampler1 = SamplerAgeRegion(data=self.data1, data_store_path=self.path)
        np.random.seed(1)
        self.assertEqual(self.sampler1.sample(len(self.sampler1.data)), ['0.0.0.1', '0.0.0.0', '0.2.0.0',
                                                                         '0.1.0.0', '0.0.1.0', '1.0.0.0'])

    def test_sample3(self):
        self.data2 = pd.DataFrame({'id': ['0.0.0.0', '0.0.0.1', '0.0.0.2',
                                          '0.0.0.3', '0.0.0.4', '0.0.0.5'],
                                  'age': [1, 2, 2, 2, 1, 0]})
        self.sampler2 = SamplerAgeRegion(data=self.data2, data_store_path=self.path)
        np.random.seed(1)
        self.assertEqual(self.sampler2.sample(4, household_criterion=True, household_threshold=4),
                         ['0.0.0.0', '0.0.0.1', '0.0.0.3', '0.0.0.2'])
        with self.assertRaises(ValueError):
            self.sampler2.sample(len(self.sampler2.data), household_criterion=True, household_threshold=1)

    def test_additional_nonresponder(self):
        self.data3 = pd.DataFrame({'id': ['0.0.0.0', '0.0.0.1', '0.0.1.0', '0.1.0.0',
                                          '0.2.0.0', '1.0.0.0'],
                                   'age': [1, 81, 45, 33, 20, 60]})
        self.sampler3 = SamplerAgeRegion(data=self.data3, data_store_path=self.path)
        expected_res = np.zeros((2, 17))
        expected_res[0, 0] = 1
        try:
            assert_array_equal(np.array(self.sampler3.additional_nonresponder(['0.0.0.0'], 1, 1)), expected_res)
        except AssertionError:
            self.fail('additional samples not generated as expected')
        try:
            assert_array_equal(np.array(self.sampler3.additional_nonresponder(['0.0.0.0'], 1, 0, 1)), expected_res)
        except AssertionError:
            self.fail('additional samples not generated as expected')
        expected_res[0, 0] = 0
        expected_res[0, 16] = 1
        try:
            assert_array_equal(np.array(self.sampler3.additional_nonresponder(['0.0.0.1'], 1, 1)), expected_res)
        except AssertionError:
            self.fail('additional samples not generated as expected')

    @patch('numpy.random.rand')
    @patch('builtins.print')
    def test_multinomial_fail(self, mock_print, mock_rand):
        afterwards_rand = [0.05] * 100
        mock_rand.side_effect = [0.02, 0.02, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] + afterwards_rand
        self.data4 = pd.DataFrame({
            'id': [
                '0.0.0.0',
                '0.0.0.1',
                '1.0.0.0',
                '1.0.0.1',
                '2.0.0.0',
                '2.0.0.1',
                '2.0.0.2',
                '2.0.0.3',
                '2.0.0.4',
                '2.0.0.5',
                '2.0.0.6',
                '2.0.0.7',
                '2.0.0.8'
            ],
            'age': [
                1,
                6,
                1,
                6,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1
            ]
        })
        self.sampler4 = SamplerAgeRegion(data=self.data4, data_store_path=self.path, num_age_group=2)
        prob = [
            0.03,
            0.06,
            0.03,
            0.06,
            0.82,
            0
        ]
        self.sampler4.multinomial_draw(10, prob)
        mock_print.assert_called_with('Multinomial draw failed for the first time. Retry with loose caps.')

    @patch('numpy.random.rand')
    @patch('builtins.print')
    def test_multinomial_fail2(self, mock_print, mock_rand):
        afterwards_rand = [0.05] * 100
        mock_rand.side_effect = [0.005, 0.005, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5] + afterwards_rand
        self.data5 = pd.DataFrame({
            'id': [
                '0.0.0.0',
                '0.0.0.1',
                '0.0.0.2',
                '0.0.0.3',
                '0.0.0.4',
                '0.0.0.5',
                '0.0.0.6',
                '0.0.0.7',
                '0.0.0.8',
                '0.0.0.9',
                '0.0.0.10',
                '1.0.0.0',
                '1.0.0.1'
            ],
            'age': [
                1,
                6,
                11,
                11,
                11,
                11,
                11,
                11,
                11,
                11,
                11,
                1,
                6
            ]
        })
        self.sampler5 = SamplerAgeRegion(data=self.data5, data_store_path=self.path, num_age_group=3)
        prob = [
            0.01,
            0.01,
            0.86,
            0.06,
            0.06,
            0
        ]
        self.sampler5.multinomial_draw(10, prob)
        mock_print.assert_called_with('Multinomial draw failed for the first time. Retry with loose caps.')

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
