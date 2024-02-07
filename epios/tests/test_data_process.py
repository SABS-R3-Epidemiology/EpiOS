import pandas as pd
import unittest
from unittest import TestCase
from data_process import DataProcess
import os
import json
from pandas.testing import assert_frame_equal


class TestDataProcess(TestCase):

    def setUp(self) -> None:
        '''
        Set up the testing files
        This function includes input data and expected result

        '''
        self.path = './testing_dataprocess/'
        try:
            os.mkdir(self.path[2:-1])
        except FileExistsError:
            raise FileExistsError('Directory already exists, terminated not to overwrite anything!')
        self.data = pd.DataFrame({'id': ['0.0.0.0', '0.0.0.1', '0.0.1.0', '0.1.0.0',
                                         '0.2.0.0', '1.0.0.0'], 'age': [1, 101, 45, 33, 20, 60]})
        self.processor = DataProcess(self.data, path=self.path)
        self.expected_json = [1 / 6, 0.0, 0.0, 0.0, 1 / 6, 0.0, 1 / 6, 0.0, 0.0,
                              1 / 6, 0.0, 0.0, 1 / 6, 0.0, 0.0, 0.0, 1 / 6]
        self.expected_df_microcell = pd.DataFrame({'cell': [0, 0, 0, 0, 1],
                                                   'microcell': [0, 0, 1, 2, 0],
                                                   'household': [0, 1, 0, 0, 0],
                                                   'Susceptible': [2, 1, 1, 1, 1]})
        self.expected_df_population = pd.DataFrame({'id': ['0.0.0.0', '0.0.0.1',
                                                           '0.0.1.0', '0.1.0.0',
                                                           '0.2.0.0', '1.0.0.0'],
                                                    'age': [1, 101, 45, 33, 20, 60],
                                                    'cell': [0, 0, 0, 0, 0, 1],
                                                    'microcell': [0, 0, 0, 1, 2, 0],
                                                    'household': [0, 0, 1, 0, 0, 0]})

    def test_data_process_ageregion(self):
        self.processor.gen_ageinfo = True
        self.processor.gen_geoinfo = True
        self.processor.pre_process(path=self.path, num_age_group=17, age_group_width=5)
        self.assertTrue(os.path.exists(self.path + 'pop_dist.json'))
        self.assertTrue(os.path.exists(self.path + 'data.csv'))
        self.assertTrue(os.path.exists(self.path + 'microcells.csv'))
        with open(self.path + 'pop_dist.json', 'r') as f:
            data = json.load(f)
            self.assertEqual(data, self.expected_json)
        df_microcell = pd.read_csv(self.path + 'microcells.csv')
        try:
            assert_frame_equal(df_microcell, self.expected_df_microcell)
        except AssertionError:
            self.fail('microcells.csv is not generated as expected')
        df_population = pd.read_csv(self.path + 'data.csv')
        try:
            assert_frame_equal(df_population, self.expected_df_population)
        except AssertionError:
            self.fail('data.csv is not generated as expected')
        if os.path.exists(self.path + 'pop_dist.json'):
            os.remove(self.path + 'pop_dist.json')
        if os.path.exists(self.path + 'microcells.csv'):
            os.remove(self.path + 'microcells.csv')
        if os.path.exists(self.path + 'data.csv'):
            os.remove(self.path + 'data.csv')

    def test_data_process_age(self):
        self.processor.gen_ageinfo = True
        self.processor.gen_geoinfo = False
        self.processor.pre_process(path=self.path, num_age_group=17, age_group_width=5)
        self.assertTrue(os.path.exists(self.path + 'pop_dist.json'))
        self.assertTrue(os.path.exists(self.path + 'data.csv'))
        self.assertFalse(os.path.exists(self.path + 'microcells.csv'))
        with open(self.path + 'pop_dist.json', 'r') as f:
            data = json.load(f)
            self.assertEqual(data, self.expected_json)
        df_population = pd.read_csv(self.path + 'data.csv')
        try:
            assert_frame_equal(df_population, self.data)
        except AssertionError:
            self.fail('data.csv is not generated as expected')
        if os.path.exists(self.path + 'pop_dist.json'):
            os.remove(self.path + 'pop_dist.json')
        if os.path.exists(self.path + 'microcells.csv'):
            os.remove(self.path + 'microcells.csv')
        if os.path.exists(self.path + 'data.csv'):
            os.remove(self.path + 'data.csv')

    def test_data_process_region(self):
        self.expected_df_population = pd.DataFrame({'id': ['0.0.0.0', '0.0.0.1',
                                                           '0.0.1.0', '0.1.0.0',
                                                           '0.2.0.0', '1.0.0.0'],
                                                    'cell': [0, 0, 0, 0, 0, 1],
                                                    'microcell': [0, 0, 0, 1, 2, 0],
                                                    'household': [0, 0, 1, 0, 0, 0]})
        self.processor.gen_ageinfo = False
        self.processor.gen_geoinfo = True
        self.processor.pre_process(path=self.path)
        self.assertFalse(os.path.exists(self.path + 'pop_dist.json'))
        self.assertTrue(os.path.exists(self.path + 'data.csv'))
        self.assertTrue(os.path.exists(self.path + 'microcells.csv'))
        df_microcell = pd.read_csv(self.path + 'microcells.csv')
        try:
            assert_frame_equal(df_microcell, self.expected_df_microcell)
        except AssertionError:
            self.fail('microcells.csv is not generated as expected')
        df_population = pd.read_csv(self.path + 'data.csv')
        try:
            assert_frame_equal(df_population, self.expected_df_population)
        except AssertionError:
            self.fail('data.csv is not generated as expected')

    def test__init__(self):
        self.processor_init = DataProcess(self.data, path=self.path, mode='AgeRegion',
                                          age_group_width=5, num_age_group=17)
        self.assertTrue(self.processor_init.gen_ageinfo)
        self.assertTrue(self.processor_init.gen_geoinfo)
        self.processor_init = DataProcess(self.data, path=self.path, mode='Base')
        self.assertFalse(self.processor_init.gen_ageinfo)
        self.assertFalse(self.processor_init.gen_geoinfo)
        self.processor_init = DataProcess(self.data, path=self.path, mode='Age',
                                          age_group_width=5, num_age_group=17)
        self.assertTrue(self.processor_init.gen_ageinfo)
        self.assertFalse(self.processor_init.gen_geoinfo)
        self.processor_init = DataProcess(self.data, path=self.path, mode='Region')
        self.assertFalse(self.processor_init.gen_ageinfo)
        self.assertTrue(self.processor_init.gen_geoinfo)

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
