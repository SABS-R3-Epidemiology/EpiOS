import numpy as np
import pandas as pd
import unittest
from unittest import TestCase
from epios.sampling_maker import SamplingMaker
import os


class TestSM(TestCase):

    def test_positive(self):
        self.assertEqual(SamplingMaker(threshold=1).testresult(1.1),'Positive')
        for a in ['InfectASympt', 'InfectMild', 'InfectGP', 'InfectHosp', 'InfectICU', 'InfectICURecov']:
            self.assertEqual(SamplingMaker().testresult(a),'Positive')

    def test_negative(self):
        self.assertEqual(SamplingMaker(threshold=1).testresult(0.9),'Negative')
        for a in ['Susceptible', 'Exposed', 'Recovered', 'Dead', 'Vaccinated']:
            self.assertEqual(SamplingMaker().testresult(a),'Negative')

    def test_false_positive(self):
        self.assertEqual(SamplingMaker(threshold=1, false_positive=1).testresult(0.9),'Positive')
        for a in ['Susceptible', 'Exposed', 'Recovered', 'Dead', 'Vaccinated']:
            self.assertEqual(SamplingMaker(false_positive=1).testresult(a),'Positive')

    def test_false_negative(self):
        self.assertEqual(SamplingMaker(threshold=1,false_negative=1).testresult(1.1),'Negative')
        for a in ['InfectASympt', 'InfectMild', 'InfectGP', 'InfectHosp', 'InfectICU', 'InfectICURecov']:
            self.assertEqual(SamplingMaker(false_negative=1).testresult(a),'Negative')

    def test_nonresponders(self):
        self.assertEqual(SamplingMaker(nonresprate=1).testresult(None),'NonResponder')

    def test__call__(self):
        t = [0,2]
        X=SamplingMaker(TheLoads=pd.DataFrame({0:[0,0,1],1:[1,1,0],2:[2,2,2]}))
        self.assertEqual(X(t,[0,1]),pd.DataFrame({0:['False','True'],1:['True','False']}))
        X=SamplingMaker(TheLoads=pd.DataFrame({0:['Susceptible','Susceptible','InfectASympt'],1:['InfectASympt','InfectASympt','Susceptible'],2:[2,2,2]}))
        self.assertEqual(X(t,[[0,1],[1,0]]),[pd.DataFrame({0:'False',1:'True'}),pd.DataFrame({0:'True',1:'False'})])