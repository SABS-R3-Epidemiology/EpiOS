import pandas
from pyEpiabm.property.infection_status import InfectionStatus
from sampling.age_region import NonResponder
from numpy.random import binomial

class SamplingMaker(NonResponder):
    '''Class to return the results of sampling

    '''

    def __init__(self, HYPER_PARAM, THE_DATA):
        if 'geoinfo' not in HYPER_PARAM:
            raise 'plese specify geoinfo'
        if 'data' not in HYPER_PARAM:
            raise 'plese specify data'
        if 'geoinfo' not in HYPER_PARAM:
            raise 'plese specify nonRespRate'
        super().__init__(self, THE_DATA.pop('geoinfo_path'), THE_DATA.pop('data'), HYPER_PARAM.pop('nonRespRate'))
        self.sampling_times=HYPER_PARAM.pop('sampling_times')
        self.sample_size=HYPER_PARAM.pop('sample_size')
        self.HYPER_PARAM=HYPER_PARAM
        self.THE_DATA_STATUSES=THE_DATA.pop('STATUSES')
        self.THE_DATA_VIRLOADS=THE_DATA.pop('VIRLOADS')
        self.set_profile()

    def testresult(self,load):
        if load>self.HYPER_PARAM['threshold']:
            return binomial(1, 1 - HYPER_PARAM['false_negative'])

    def sample(self):
        if 'nonresponders' in self.HYPER_PARAM:
            nonresponders=self.HYPER_PARAM['nonresponders']
            if 'sampling_percentage' not in nonresponders:
                nonresponders['sampling_percentage']=0.1
            if 'proportion' not in nonresponders:
                nonresponders['proportion']=0.01
            if 'threshold' not in nonresponders:
                nonresponders['threshold']=None
            return super().sample(sample_size=self.sample_size,
                                            additional_sample=self.additional_sample(sampling_percentage=nonresponders['sampling_percentage'],
                                                                                     proportion=nonresponders['proportion'],
                                                                                     threshold=nonresponders['threshold']))
        else:
            return super().sample(sample_size=self.HYPER_PARAM['sample_size'])

    def results_of_sampling(self):
        # REMEMBER TO SWITCH THE PARAMETERS OF DataFrame.loc PROPERLY
        if self.HYPER_PARAM['keep_track'] and self.HYPER_PARAM['ifectiousness_profile']:
            STATUSES = self.THE_DATA_VIRLOADS.loc[self.sampling_times,self.sample()]
            return STATUSES.apply(lambda x)
        else:
            # this computes a list of dataframes with only one row/column
            STATUSES = map(lambda t:self.sample().loc[t,self.sample()],self.sampling_times)
            # this applies res to all of them
            return list(map(lambda x: x.apply(self.res),STATUSES))