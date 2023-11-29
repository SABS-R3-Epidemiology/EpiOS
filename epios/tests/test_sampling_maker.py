import unittest
import pandas
from math import exp
from numpy.random import binomial
from pyEpiabm.property.infection_status import InfectionStatus
from sampling.sampling_maker import SamplingMaker

class TestSamplingMaking():

    def test_results_of_sampling(self):
        
        self.THE_DATA=pandas.DataFrame()
        
        def ifectiousness_profile(time_passed, status):
            return (time_passed,status)
        
        cell=[0,0,1,1]
        microcell=[0,1,0,1]
        location_x=[0.0,0.0,1.0,1.0]
        location_y=[0.0,1.0,0.0,1.0]
        household_number=[2,2,2,2]
        Subsceptible=[4,4,4,4]
        geoinfo=pandas.DataFrame({'cell':cell,
                                  'microcell':microcell,
                                  'location_x':location_x,
                                  'location_y':location_y,
                                  'household_number':household_number,
                                  'Subsceptible':Subsceptible})
        geoinfo_path=''
        geoinfo.to_csv(geoinfo_path)
        cell=[0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
        microcell=[0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1]
        household=[0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]
        age=[20,40,20,40,20,40,20,40,20,40,20,40,20,40,20,40]
        data=pandas.DataFrame({'ID':list(range(16)),
                               'cell':cell,
                               'microcell':microcell,
                               'household':household,
                               'age':age})
        self.HYPER_PARAM={'ifectiousness_profile':ifectiousness_profile,
                          'sampling_times':[0,1],
                          'sample_size':16,
                          'geoinfo_path':geoinfo_path,
                          'data':data}
        A=list(map(lambda x:InfeInfo(x[0],x[1]),zip(range(16),range(16))))
        B=list(map(lambda x:InfeInfo(x[0],x[1]),zip(range(16),range(16))))
        THE_DATA=pandas.DataFrame({0:list(range(16)),1:list(range(16))})
        samplingmaker=SamplingMaker(self.HYPER_PARAM|{'keep_track':True})
        results_of_sampling=samplingmaker.results_of_sampling(THE_DATA)
        assert(results_of_sampling==THE_DATA.apply(lambda x: (x.time_of_status_change, x.status)))
        samplingmaker=SamplingMaker(self.HYPER_PARAM|{'keep_track':False})
        samplingmaker.results_of_sampling(THE_DATA)

