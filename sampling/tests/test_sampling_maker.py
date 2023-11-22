import unittest
import pandas
from math import exp
from numpy.random import binomial
from pyEpiabm.property.infection_status import InfectionStatus
from sampling.sampling_maker import SamplingMaker

class TestSamplingMaking():

    def test_results_of_sampling(self):
        
        self.THE_DATA=pandas.DataFrame()
        
        def ifectiousness_profile(time_passed ,status):
            if status == InfectionStatus.InfectAsympt:
                p = time_passed*exp(1 - time_passed)
            elif status == InfectionStatus.InfectMild:
                p = time_passed*exp(1 - time_passed)
            elif status == InfectionStatus.InfectGP:
                p = time_passed*exp(1 - time_passed)
            elif status == InfectionStatus.InfectHosp:
                p = time_passed*exp(1 - time_passed)
            elif status == InfectionStatus.InfectICU:
                p = time_passed*exp(1 - time_passed)
            elif status == InfectionStatus.InfectRecov:
                p = time_passed*exp(1 - time_passed)
            else: return False
            return bool(binomial(1,p))
        
        sampling_times=[1,2,3]
        self.HYPER_PARAM={'ifectiousness_profile':ifectiousness_profile,'sampling_times':sampling_times,'geoinfo','data','nonRespRate'}
