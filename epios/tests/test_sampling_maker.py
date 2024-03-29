import pandas as pd
from unittest import TestCase
from sampling_maker import SamplingMaker


class TestSM(TestCase):

    def test_positive(self):
        self.assertEqual(SamplingMaker(threshold=1).testresult(1.1), 'Positive')
        for a in [3, 4, 5, 6, 7, 8]:
            self.assertEqual(SamplingMaker().testresult(a), 'Positive')

    def test_negative(self):
        self.assertEqual(SamplingMaker(threshold=1).testresult(0.9), 'Negative')
        for a in [1, 2, 9, 10, 11]:
            self.assertEqual(SamplingMaker().testresult(a), 'Negative')

    def test_false_positive(self):
        self.assertEqual(SamplingMaker(threshold=1, false_positive=1).testresult(0.9), 'Positive')
        for a in [1, 2, 9, 10, 11]:
            self.assertEqual(SamplingMaker(false_positive=1).testresult(a), 'Positive')

    def test_false_negative(self):
        self.assertEqual(SamplingMaker(threshold=1, false_negative=1).testresult(1.1), 'Negative')
        for a in [3, 4, 5, 6, 7, 8]:
            self.assertEqual(SamplingMaker(false_negative=1).testresult(a), 'Negative')

    def test_nonresponders(self):
        self.assertEqual(SamplingMaker(non_resp_rate=1).testresult(None), 'NonResponder')

    def test__call__(self):
        t = [0, 2]
        X = SamplingMaker(keeptrack=True, threshold=0.5, TheData=pd.DataFrame({0: [0, 0, 1],
                                                                               1: [1, 1, 0], 2: [2, 2, 2]}))
        self.assertFalse((X(t, [0, 1]) != pd.DataFrame({0: ['Negative', 'Positive'],
                                                        1: ['Positive', 'Negative']}, index=[0, 2])).any(axis=None))
        X = SamplingMaker(TheData=pd.DataFrame({0: [1, 1, 3],
                                                1: [3, 3, 1], 2: [2, 2, 2]}))
        self.assertFalse((X(t, [[0, 1], [0, 1]])[0] != pd.DataFrame({0: 'Negative',
                                                                     1: 'Positive'}, index=[0])).any(axis=None))
        self.assertFalse((X(t, [[0, 1], [0, 1]])[1] != pd.DataFrame({0: 'Positive',
                                                                     1: 'Negative'}, index=[2])).any(axis=None))
