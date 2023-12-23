import numpy as np
import pandas as pd
import unittest
from unittest import TestCase
from unittest.mock import patch
from post_process import PostProcess
import os
import sys
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
        self.path = './testing_post_process_mod/'
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
        self.processor = PostProcess(self.demo_data, self.time_data)
        self.demo_data2 = pd.DataFrame({'ID': ['0.0.0.0', '0.0.0.1', '0.0.0.2',
                                               '0.0.0.3', '0.0.0.4', '0.0.0.5'],
                                        'age': [1, 2, 1, 1, 1, 1]})
        self.time_data2 = pd.DataFrame({'time': [0, 1, 2, 3, 4, 5],
                                        '0.0.0.0': ['InfectASympt'] * 6,
                                        '0.0.0.1': ['InfectASympt'] * 6,
                                        '0.0.0.2': ['InfectASympt'] * 6,
                                        '0.0.0.3': ['InfectASympt'] * 6,
                                        '0.0.0.4': ['InfectASympt'] * 6,
                                        '0.0.0.5': ['InfectASympt'] * 6})
        self.processor_non_responder = PostProcess(self.demo_data2, self.time_data2)
        self.demo_data3 = pd.DataFrame({'ID': ['0.0.0.0', '0.0.0.1', '0.0.0.2',
                                               '1.0.0.0', '1.0.0.1', '1.0.0.2'],
                                        'age': [1, 2, 1, 1, 1, 1]})
        self.time_data3 = pd.DataFrame({'time': [0, 1, 2, 3, 4, 5],
                                        '0.0.0.0': ['InfectASympt'] * 6,
                                        '0.0.0.1': ['InfectASympt'] * 6,
                                        '0.0.0.2': ['InfectASympt'] * 6,
                                        '1.0.0.0': ['InfectASympt'] * 6,
                                        '1.0.0.1': ['InfectASympt'] * 6,
                                        '1.0.0.2': ['InfectASympt'] * 6})
        self.processor_non_responder2 = PostProcess(self.demo_data3, self.time_data3)

    def test__init__(self):
        try:
            assert_frame_equal(self.processor.demo_data, pd.DataFrame({'ID': ['0.0.0.0', '0.0.0.1', '0.0.1.0',
                                                                              '0.1.0.0', '0.2.0.0', '1.0.0.0'],
                                                                       'age': [1, 81, 45, 33, 20, 60]}))
            assert_frame_equal(self.processor.time_data, pd.DataFrame({'time': [0, 1, 2, 3, 4, 5],
                                                                       '0.0.0.0': ['InfectASympt'] * 6,
                                                                       '0.0.0.1': [0, 0, 0, 'InfectASympt',
                                                                                   'InfectASympt', 'InfectASympt'],
                                                                       '0.0.1.0': [0, 0, 'InfectASympt',
                                                                                   'InfectASympt', 'InfectASympt',
                                                                                   'InfectASympt'],
                                                                       '0.1.0.0': [0, 0, 'InfectASympt',
                                                                                   'InfectASympt',
                                                                                   'InfectASympt', 'InfectASympt'],
                                                                       '0.2.0.0': [0, 'InfectASympt', 'InfectASympt',
                                                                                   'InfectASympt', 'InfectASympt',
                                                                                   'InfectASympt'],
                                                                       '1.0.0.0': [0, 0, 0, 0, 0, 'InfectASympt']}))
        except AssertionError:
            self.fail('init function error')

    def test_sampled_result_s(self):
        kwargs = {
            'sample_strategy': 'Same',
            'gen_plot': True,
            'saving_path_sampling': self.path + 'sample.png'
        }
        res = self.processor('AgeRegion', 6, [0, 1, 2, 3, 4, 5], comparison=False, data_store_path=self.path, **kwargs)
        self.assertAlmostEqual(res, [[0, 1, 2, 3, 4, 5], [1 / 6, 2 / 6, 4 / 6, 5 / 6, 5 / 6, 6 / 6]])
        assert os.path.exists(self.path + 'sample.png'), "Plot file was not saved"
        if os.path.exists(self.path + 'sample.png'):
            os.remove(self.path + 'sample.png')
        res = self.processor('Age', 6, [0, 1, 2, 3, 4, 5], comparison=False, data_store_path=self.path, **kwargs)
        self.assertAlmostEqual(res, [[0, 1, 2, 3, 4, 5], [1 / 6, 2 / 6, 4 / 6, 5 / 6, 5 / 6, 6 / 6]])
        assert os.path.exists(self.path + 'sample.png'), "Plot file was not saved"
        if os.path.exists(self.path + 'sample.png'):
            os.remove(self.path + 'sample.png')
        res = self.processor('Region', 6, [0, 1, 2, 3, 4, 5], comparison=False, data_store_path=self.path, **kwargs)
        self.assertAlmostEqual(res, [[0, 1, 2, 3, 4, 5], [1 / 6, 2 / 6, 4 / 6, 5 / 6, 5 / 6, 6 / 6]])
        assert os.path.exists(self.path + 'sample.png'), "Plot file was not saved"
        if os.path.exists(self.path + 'sample.png'):
            os.remove(self.path + 'sample.png')
        res = self.processor('Base', 6, [0, 1, 2, 3, 4, 5], comparison=False, data_store_path=self.path, **kwargs)
        self.assertAlmostEqual(res, [[0, 1, 2, 3, 4, 5], [1 / 6, 2 / 6, 4 / 6, 5 / 6, 5 / 6, 6 / 6]])
        assert os.path.exists(self.path + 'sample.png'), "Plot file was not saved"
        with self.assertRaises(ValueError):
            res = self.processor('a', 6, [0, 1, 2, 3, 4, 5], comparison=False, data_store_path=self.path, **kwargs)

    def test_sampled_result_r(self):
        if os.path.exists(self.path + 'sample.png'):
            os.remove(self.path + 'sample.png')
        np.random.seed(1)
        kwargs = {
            'sample_strategy': 'Random',
            'gen_plot': True,
            'saving_path_sampling': self.path + 'sample.png'
        }
        res = self.processor('AgeRegion', 6, [0, 1, 2, 3, 4, 5], comparison=False,
                             data_store_path=self.path, **kwargs)
        self.assertAlmostEqual(res, [[0, 1, 2, 3, 4, 5], [1 / 6, 2 / 6, 4 / 6, 5 / 6, 5 / 6, 6 / 6]])
        assert os.path.exists(self.path + 'sample.png'), "Plot file was not saved"
        if os.path.exists(self.path + 'sample.png'):
            os.remove(self.path + 'sample.png')
        res = self.processor('Age', 6, [0, 1, 2, 3, 4, 5], comparison=False, data_store_path=self.path, **kwargs)
        self.assertAlmostEqual(res, [[0, 1, 2, 3, 4, 5], [1 / 6, 2 / 6, 4 / 6, 5 / 6, 5 / 6, 6 / 6]])
        assert os.path.exists(self.path + 'sample.png'), "Plot file was not saved"
        if os.path.exists(self.path + 'sample.png'):
            os.remove(self.path + 'sample.png')
        res = self.processor('Region', 6, [0, 1, 2, 3, 4, 5], comparison=False,
                             data_store_path=self.path, **kwargs)
        self.assertAlmostEqual(res, [[0, 1, 2, 3, 4, 5], [1 / 6, 2 / 6, 4 / 6, 5 / 6, 5 / 6, 6 / 6]])
        assert os.path.exists(self.path + 'sample.png'), "Plot file was not saved"
        if os.path.exists(self.path + 'sample.png'):
            os.remove(self.path + 'sample.png')
        res = self.processor('Base', 6, [0, 1, 2, 3, 4, 5], comparison=False, data_store_path=self.path, **kwargs)
        self.assertAlmostEqual(res, [[0, 1, 2, 3, 4, 5], [1 / 6, 2 / 6, 4 / 6, 5 / 6, 5 / 6, 6 / 6]])
        assert os.path.exists(self.path + 'sample.png'), "Plot file was not saved"

    def test_sampled_non_responder(self):
        with self.assertRaises(ValueError):
            self.processor('Age', 6, [0, 1, 2, 3, 4, 5], comparison=False,
                           data_store_path=self.path, non_responder=True, nonresprate=0)
        with self.assertRaises(ValueError):
            self.processor('Base', 6, [0, 1, 2, 3, 4, 5], comparison=False,
                           data_store_path=self.path, non_responder=True, nonresprate=0)
        with self.assertRaises(ValueError):
            self.processor('a', 6, [0, 1, 2, 3, 4, 5], comparison=False,
                           data_store_path=self.path, non_responder=True, nonresprate=0)
        with self.assertRaises(ValueError):
            self.processor('AgeRegion', 6, [0, 1, 2, 3, 4, 5], comparison=False,
                           data_store_path=self.path, non_responder=True)
        np.random.seed(1)
        kwargs = {
            'gen_plot': True,
            'saving_path_sampling': self.path + 'sample_nonResp.png'
        }
        res = self.processor('AgeRegion', 6, [0, 1, 2, 3, 4, 5], comparison=False, non_responder=True,
                             data_store_path=self.path, nonresprate=0.1, **kwargs)
        self.assertAlmostEqual(res, [[0, 1, 2, 3, 4, 5], [1 / 6, 0.4, 2 / 3, 5 / 6, 1, 1]])
        assert os.path.exists(self.path + 'sample_nonResp.png'), "Plot file was not saved"
        res = self.processor('AgeRegion', 6, [0, 1, 2, 3, 4, 5], comparison=False, non_responder=True,
                             data_store_path=self.path, nonresprate=0.9, **kwargs)
        self.assertEqual(res, [[0, 1, 2, 3, 4, 5], [0.0, np.nan, np.nan, 1.0, np.nan, np.nan]])
        np.random.seed(1)
        kwargs['proportion'] = 1
        kwargs['sampling_percentage'] = 1
        res = self.processor_non_responder('AgeRegion', 3, [0, 1, 2, 3, 4, 5], comparison=False, non_responder=True,
                                           data_store_path=self.path, nonresprate=0.9, **kwargs)
        self.assertEqual(res, [[0, 1, 2, 3, 4, 5], [1.0, 1.0, 1.0, 1.0, 1.0, np.nan]])
        np.random.seed(1)
        res = self.processor_non_responder2('AgeRegion', 4, [0, 1, 2, 3, 4, 5], comparison=False, non_responder=True,
                                            data_store_path=self.path, nonresprate=0.8, **kwargs)
        self.assertEqual(res, [[0, 1, 2, 3, 4, 5], [1.0, 1.0, np.nan, 1.0, 1.0, 1.0]])
        kwargs = {
            'gen_plot': True,
            'saving_path_sampling': self.path + 'sample_nonResp.png'
        }
        res = self.processor('Region', 6, [0, 1, 2, 3, 4, 5], comparison=False, non_responder=True,
                             data_store_path=self.path, nonresprate=0.1, **kwargs)
        self.assertEqual(res, [[0, 1, 2, 3, 4, 5], [0.2, 1 / 3, 2 / 3, 0.8, 0.8, 1]])
        assert os.path.exists(self.path + 'sample_nonResp.png'), "Plot file was not saved"
        res = self.processor('Region', 6, [0, 1, 2, 3, 4, 5], comparison=False, non_responder=True,
                             data_store_path=self.path, nonresprate=0.9, **kwargs)
        self.assertEqual(res, [[0, 1, 2, 3, 4, 5], [np.nan, 0.5, np.nan, 1.0, np.nan, np.nan]])
        np.random.seed(1)
        kwargs['proportion'] = 1
        kwargs['sampling_percentage'] = 1
        res = self.processor_non_responder('Region', 3, [0, 1, 2, 3, 4, 5], comparison=False, non_responder=True,
                                           data_store_path=self.path, nonresprate=0.9, **kwargs)
        self.assertEqual(res, [[0, 1, 2, 3, 4, 5], [1.0, 1.0, 1.0, 1.0, 1.0, np.nan]])
        np.random.seed(1)
        res = self.processor_non_responder2('Region', 4, [0, 1, 2, 3, 4, 5], comparison=False, non_responder=True,
                                            data_store_path=self.path, nonresprate=0.9, **kwargs)
        self.assertEqual(res, [[0, 1, 2, 3, 4, 5], [1.0, 1.0, 1.0, 1.0, 1.0, np.nan]])

    def test_compare(self):
        kwargs = {
            'sample_strategy': 'Same',
            'saving_path_compare': self.path + 'compare.png',
            'gen_plot': True
        }
        _, diff = self.processor('AgeRegion', 6, [0, 1, 2, 3, 4, 5], comparison=True,
                                 data_store_path=self.path, **kwargs)
        self.assertEqual(list(diff), [0, 0, 0, 0, 0, 0])
        assert os.path.exists(self.path + 'compare.png'), "Plot file was not saved"
        if os.path.exists(self.path + 'compare.png'):
            os.remove(self.path + 'compare.png')
        kwargs = {
            'saving_path_compare': self.path + 'compare.png',
            'gen_plot': True
        }
        _, diff = self.processor('AgeRegion', 6, [0, 1, 2, 3, 4, 5], comparison=True,
                                 non_responder=True, nonresprate=0,
                                 data_store_path=self.path, **kwargs)
        self.assertEqual(list(diff), [0, 0, 0, 0, 0, 0])
        assert os.path.exists(self.path + 'compare.png'), "Plot file was not saved"

    def test_best_method_error(self):
        with self.assertRaises(ValueError):
            self.processor.best_method(
                methods=[
                    'Base-Same'
                ],
                sample_size=3,
                hyperparameter_autotune=False,
                non_responder=False,
                sampling_interval=1,
                iteration=1,
                nonresprate=0.1,
                metric='a',
                parallel_computation=False,
            )
        with self.assertRaises(ValueError):
            self.processor.best_method(
                methods=[
                    'a'
                ],
                sample_size=3,
                hyperparameter_autotune=False,
                non_responder=False,
                sampling_interval=1,
                iteration=1,
                nonresprate=0.1,
                metric='mean',
                parallel_computation=False,
            )

    @patch('builtins.print')
    def test_mock_print_normal(self, mock_print):
        kwargs = {
            'a': 1
        }
        _ = self.processor('AgeRegion', 6, [0, 1, 2, 3, 4, 5],
                           data_store_path=self.path, comparison=False, **kwargs)
        mock_print.assert_called_with("The following parameters provided are not used: a")

    @patch('builtins.print')
    def test_mock_print_non_responder(self, mock_print):
        kwargs = {
            'a': 1
        }
        _ = self.processor('AgeRegion', 6, [0, 1, 2, 3, 4, 5], comparison=False,
                           non_responder=True, data_store_path=self.path, nonresprate=0, **kwargs)
        mock_print.assert_called_with("The following parameters provided are not used: a")

    @patch('builtins.print')
    def test_mock_print_best_method_parallel(self, mock_print):
        remove = 0
        kwargs = {
            'num_age_group_range': [17],
            'age_group_width_range': [5],
            'sampling_percentage_range': [0.1],
            'proportion_range': [0.01],
            'threshold_range': [1],
            'good': 1
        }
        main_module_path = os.path.abspath(sys.modules['__main__'].__file__)
        dir_name = os.path.dirname(main_module_path) + '/'
        if os.path.exists(dir_name + 'temp_'):
            remove = 1
        else:
            os.mkdir(dir_name + 'temp_')
        self.processor.best_method(
            methods=[
                'AgeRegion',
                'a'
            ],
            sample_size=3,
            hyperparameter_autotune=False,
            non_responder=False,
            sampling_interval=1,
            iteration=10,
            nonresprate=0.1,
            metric='max',
            parallel_computation=True,
            **kwargs
        )
        mock_print.assert_called()
        if remove == 0:
            os.rmdir(dir_name + 'temp_')

    @patch('builtins.print')
    def test_mock_print_best_method_nonparallel(self, mock_print):
        np.random.seed(1)
        kwargs = {
            'num_age_group': 17,
            'num_age_group_range': [17],
            # 'age_group_width_range': [5],
            'sampling_percentage_range': [0.1],
            'proportion_range': [0.01],
            'threshold_range': [1]
        }
        self.processor.best_method(
            methods=[
                'Base-Same',
                'Base-Random',
                'Age-Same',
                'AgeRegion-Random'
            ],
            sample_size=3,
            hyperparameter_autotune=True,
            non_responder=False,
            sampling_interval=1,
            iteration=1,
            nonresprate=0.1,
            metric='mean',
            parallel_computation=False,
            **kwargs
        )
        mock_print.assert_called()
        np.random.seed(1)
        kwargs = {
            'num_age_group': 17,
            'num_age_group_range': [17],
            # 'age_group_width_range': [5],
            # 'sampling_percentage_range': [0.1],
            'proportion_range': [0.01],
            'threshold_range': [1]
        }
        self.processor.best_method(
            methods=[
                'Base-Same',
                'Base-Random',
                'Region-Random',
                'AgeRegion-Random'
            ],
            sample_size=3,
            hyperparameter_autotune=True,
            non_responder=True,
            sampling_interval=1,
            iteration=1,
            nonresprate=0.1,
            metric='mean',
            parallel_computation=False,
            **kwargs
        )
        mock_print.assert_called()
        np.random.seed(1)
        kwargs = {
            'num_age_group': 17,
            'num_age_group_range': [17],
            # 'age_group_width_range': [5],
            'sampling_percentage_range': [0.1],
            'proportion_range': [0.01],
            'threshold_range': [1]
        }
        self.processor.best_method(
            methods=[
                'Base-Same',
                'AgeRegion-Random'
            ],
            sample_size=3,
            hyperparameter_autotune=False,
            non_responder=False,
            sampling_interval=1,
            iteration=1,
            nonresprate=0.1,
            metric='mean',
            parallel_computation=False,
            **kwargs
        )
        mock_print.assert_called()
        np.random.seed(1)
        self.processor.best_method(
            methods=[
                'Region-Random',
                'AgeRegion-Random'
            ],
            sample_size=3,
            hyperparameter_autotune=False,
            non_responder=True,
            sampling_interval=1,
            iteration=1,
            nonresprate=0.1,
            metric='max',
            parallel_computation=False,
            **kwargs
        )
        mock_print.assert_called()

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
            if os.path.exists(self.path + 'sample_nonResp.png'):
                os.remove(self.path + 'sample_nonResp.png')
            if os.path.exists(self.path + 'compare.png'):
                os.remove(self.path + 'compare.png')
            os.rmdir(self.path)


if __name__ == '__main__':

    unittest.main()
