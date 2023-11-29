import numpy as np
import pandas as pd
import unittest
from unittest import TestCase
from non_responder_function import additional_nonresponder
from numpy.testing import assert_array_equal


class TestDataProcess(TestCase):

    def setUp(self) -> None:
        '''
        Set up the unit test data
        Firstly use the DataProcess class to generate input files
        Secondly construct the NonResponder class

        '''
        self.data = pd.DataFrame({'ID': ['0.0.0.0', '0.0.0.1', '0.0.1.0', '0.1.0.0',
                                         '0.2.0.0', '1.0.0.0'],
                                  'age': [1, 81, 45, 33, 20, 60],
                                  'cell': [0, 0, 0, 0, 0, 1],
                                  'microcell': [0, 0, 0, 1, 2, 0],
                                  'household': [0, 0, 1, 0, 0, 0]})

    def test_additional_sample(self):
        expected_res = np.zeros((2, 17))
        expected_res[0, 0] = 1
        try:
            assert_array_equal(np.array(additional_nonresponder(self.data, ['0.0.0.0'], 2, 17, 1, 1)), expected_res)
        except:
            self.fail('additional samples not generated as expected')
        try:
            assert_array_equal(np.array(additional_nonresponder(self.data, ['0.0.0.0'], 2, 17, 1, 0, 1)), expected_res)
        except:
            self.fail('additional samples not generated as expected')


if __name__ == '__main__':

    unittest.main()
