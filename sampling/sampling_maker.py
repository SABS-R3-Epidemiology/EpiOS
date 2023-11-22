import pandas
from pyEpiabm.property.infection_status import InfectionStatus
from sampling.age_region import NonResponder

class SamplingMaker(NonResponder):
    '''Class to return the results of sampling

    '''
    
    def __init__(self, HYPER_PARAM):
        if 'geoinfo' not in HYPER_PARAM:
            raise 'plese specify geoinfo'
        if 'data' not in HYPER_PARAM:
            raise 'plese specify data'
        if 'geoinfo' not in HYPER_PARAM:
            raise 'plese specify nonRespRate'
        super().__init__(self, HYPER_PARAM.pop('geoinfo'), HYPER_PARAM.pop('data'), HYPER_PARAM.pop('nonRespRate'))
        self.sampling_times=HYPER_PARAM.pop('sampling_times')
        self.HYPER_PARAM=HYPER_PARAM
        self.set_profile()

    def set_profile(self):
        if 'ifectiousness_profile' in self.HYPER_PARAM:
            # defining the result given a status
            def res(entry):
                self.HYPER_PARAM['ifectiousness_profile'](entry.current_time - entry.time_of_status_change ,entry.status)
        else:
            # defining the result given a status
            recognised_statuses=[InfectionStatus.InfectAsympt,
                                 InfectionStatus.InfectMild,
                                 InfectionStatus.InfectGP,
                                 InfectionStatus.InfectHosp,
                                 InfectionStatus.InfectICU,
                                 InfectionStatus.InfectRecov]
            def res(entry):
                if entry.status in self.recognised_statuses:
                    return True
                else:
                    return False
        self.res=res


    def sample(self,THE_DATA):
        if 'nonresponders' in self.HYPER_PARAM:
            nonresponders=self.HYPER_PARAM['nonresponders']
            if 'sampling_percentage' not in nonresponders:
                nonresponders['sampling_percentage']=0.1
            if 'proportion' not in nonresponders:
                nonresponders['proportion']=0.01
            if 'threshold' not in nonresponders:
                nonresponders['threshold']=None
            return super().sample(sample_size=self.HYPER_PARAM['sample_size'],
                                            additional_sample=self.additional_sample(sampling_percentage=nonresponders['sampling_percentage'],
                                                                                     proportion=nonresponders['proportion'],
                                                                                     threshold=nonresponders['threshold']))
        else:
            return super().sample(sample_size=self.HYPER_PARAM['sample_size'])

    def results_of_sampling(self,THE_DATA):
        #  REMEMBER TO SWITCH THE PARAMETERS OF DataFrame.loc PROPERLY
        if self.HYPER_PARAM['keep_track']:
            # this computes a DataFrame with only the selected people's statuses at the selected times
            STATUSES = self.sample(THE_DATA).loc[self.sample(),self.sampling_times]
            # this apply res to all of them
            return self.sample(THE_DATA).apply(self.res)
        else:
            # this computes a list of Dataframes with only one row/column
            STATUSES = map(lambda t:self.sample(THE_DATA).loc[self.sample(),t],self.sampling_times)
            # this apply res to all of them
            return list(map(lambda x: x.apply(self.res),STATUSES))