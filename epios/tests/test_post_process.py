import numpy as np
import pandas as pd
import unittest
from unittest import TestCase
from post_process import PostProcess
import os
# from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal


class TestDataProcess(TestCase):

    def setUp(self) -> None:
        '''
        This function set up the testing environment
        Firstly use the DataProcess class to generate input for Sampler
        Secondly construct the Sampler class

        This function include some of the expected results

        '''
        self.path = './testing_post_process/'
        try:
            os.mkdir(self.path[2:-1])
        except FileExistsError:
            raise FileExistsError('Directory already exists, terminated not to overwrite anything!')
        self.demo_data = pd.DataFrame({'ID': ['0.0.0.0', '0.0.0.1', '0.0.1.0',
                                              '0.1.0.0', '0.2.0.0', '1.0.0.0'],
                                       'age': [1, 81, 45, 33, 20, 60]})
        self.time_data = pd.DataFrame({'time': [0, 1, 2, 3, 4, 5],
                                       '0.0.0.0': ['InfectASympt'] * 6,
                                       '0.0.0.1': [0, 0, 0, 'InfectASympt', 'InfectASympt', 'InfectASympt'],
                                       '0.0.1.0': [0, 0, 'InfectASympt', 'InfectASympt', 'InfectASympt',
                                                   'InfectASympt'],
                                       '0.1.0.0': [0, 0, 'InfectASympt', 'InfectASympt', 'InfectASympt',
                                                   'InfectASympt'],
                                       '0.2.0.0': [0, 'InfectASympt', 'InfectASympt', 'InfectASympt',
                                                   'InfectASympt', 'InfectASympt'],
                                       '1.0.0.0': [0, 0, 0, 0, 0, 'InfectASympt']})
        self.processor_s = PostProcess(self.demo_data, self.time_data, 6, [0, 1, 2, 3, 4, 5], sample_strategy='same')
        self.processor_r = PostProcess(self.demo_data, self.time_data, 6, [0, 1, 2, 3, 4, 5])

    def test__init__(self):
        self.assertEqual(self.processor_r.time_sample, [0, 1, 2, 3, 4, 5])
        self.assertEqual(self.processor_r.sample_strategy, 'random')
        try:
            assert_frame_equal(self.processor_r.demo_data, pd.DataFrame({'ID': ['0.0.0.0', '0.0.0.1', '0.0.1.0',
                                                                                '0.1.0.0', '0.2.0.0', '1.0.0.0'],
                                                                         'age': [1, 81, 45, 33, 20, 60]}))
            assert_frame_equal(self.processor_r.time_data, pd.DataFrame({'time': [0, 1, 2, 3, 4, 5],
                                                                         '0.0.0.0': ['InfectASympt'] * 6,
                                                                         '0.0.0.1': [0, 0, 0, 'InfectASympt',
                                                                                     'InfectASympt', 'InfectASympt'],
                                                                         '0.0.1.0': [0, 0, 'InfectASympt',
                                                                                     'InfectASympt', 'InfectASympt',
                                                                                     'InfectASympt'],
                                                                         '0.1.0.0': [0, 0, 'InfectASympt', 'InfectASympt',
                                                                                     'InfectASympt', 'InfectASympt'],
                                                                         '0.2.0.0': [0, 'InfectASympt', 'InfectASympt',
                                                                                     'InfectASympt', 'InfectASympt',
                                                                                     'InfectASympt'],
                                                                         '1.0.0.0': [0, 0, 0, 0, 0, 'InfectASympt']}))
        except AssertionError:
            self.fail('init function error')

    def test_sampled_result_s(self):
        res = self.processor_s.sampled_result(gen_plot=True, saving_path=self.path)
        self.assertEqual(res, [[0, 1, 2, 3, 4, 5], [1, 2, 4, 5, 5, 6]])
        assert os.path.exists(self.path + 'sample.png'), "Plot file was not saved"

    def test_sampled_result_r(self):
        if os.path.exists(self.path + 'sample.png'):
            os.remove(self.path + 'sample.png')
        np.random.seed(1)
        res = self.processor_r.sampled_result(gen_plot=True, saving_path=self.path)
        self.assertEqual(res, [[0, 1, 2, 3, 4, 5], [1, 2, 4, 5, 5, 6]])
        assert os.path.exists(self.path + 'sample.png'), "Plot file was not saved"

    def test_compare(self):
        self.processor_s.sampled_result()
        diff = self.processor_s.compare(saving_path=self.path)
        self.assertEqual(list(diff), [0, 0, 0, 0, 0, 0])
        assert os.path.exists(self.path + 'compare.png'), "Plot file was not saved"

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
            if os.path.exists(self.path + 'sample.png'):
                os.remove(self.path + 'sample.png')
            if os.path.exists(self.path + 'compare.png'):
                os.remove(self.path + 'compare.png')
            os.rmdir(self.path)


if __name__ == '__main__':

    unittest.main()