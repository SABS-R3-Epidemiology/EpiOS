import pandas as pd
import unittest
from unittest import TestCase
from epios import DataProcess
import os
import json
from pandas.testing import assert_frame_equal


class TestDataProcess(TestCase):

    def setUp(self) -> None:
        self.path = './testing_files/'
        self.data = pd.DataFrame({'ID': ['0.0.0.0', '0.0.0.1', '0.0.1.0', '0.1.0.0', '0.2.0.0', '1.0.0.0'], 'age': [1, 81, 45, 33, 20, 60]})
        self.processor = DataProcess(self.data)
        self.expected_json = [1 / 6, 0.0, 0.0, 0.0, 1 / 6, 0.0, 1 / 6, 0.0, 0.0, 1 / 6, 0.0, 0.0, 1 / 6, 0.0, 0.0, 0.0, 1 / 6]
        self.expected_df_microcell = pd.DataFrame({'cell': [0, 0, 0, 0, 1], 'microcell': [0, 0, 1, 2, 0], 'household': [0, 1, 0, 0, 0], 'Susceptible': [2, 1, 1, 1, 1]})
        self.expected_df_population = pd.DataFrame({'ID': ['0.0.0.0', '0.0.0.1', '0.0.1.0', '0.1.0.0', '0.2.0.0', '1.0.0.0'], 'age': [1, 81, 45, 33, 20, 60], 'cell': [0, 0, 0, 0, 0, 1], 'microcell': [0, 0, 0, 1, 2, 0], 'household': [0, 0, 1, 0, 0, 0]})

    def test_data_process(self):
        try:
            os.mkdir(self.path[2:-1])
        except:
            raise KeyError('Directory already exists, terminated not to overwrite anything!')
        self.processor.pre_process(path=self.path)
        self.assertTrue(os.path.exists(self.path + 'pop_dist.json'))
        self.assertTrue(os.path.exists(self.path + 'data.csv'))
        self.assertTrue(os.path.exists(self.path + 'microcells.csv'))
        with open(self.path + 'pop_dist.json', 'r') as f:
            data = json.load(f)
            self.assertEqual(data, self.expected_json)
        df_microcell = pd.read_csv(self.path + 'microcells.csv')
        try:
            assert_frame_equal(df_microcell, self.expected_df_microcell)
        except:
            self.fail('microcells.csv is not generated as expected')
        df_population = pd.read_csv(self.path + 'data.csv')
        try:
            assert_frame_equal(df_population, self.expected_df_population)
        except:
            self.fail('data.csv is not generated as expected')

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
