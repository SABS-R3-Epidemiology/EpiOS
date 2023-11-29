import pandas
from pyEpiabm.property.infection_status import InfectionStatus
from epios import NonResponder, Sampler
from numpy.random import binomial

class SamplingMaker():
    '''Class to return the results of sampling

    '''

    def __init__(self, nonresprate=0, keeptrack=False, TheLoads=None, Statuses=None, false_positive=0, false_negative=0, threshold=None):
        self.recognised=['InfectASympt', 'InfectMild', 'InfectGP', 'InfectHosp', 'InfectICU', 'InfectICURecov']
        self.nonresprate=nonresprate
        self.keeptrack=keeptrack
        self.TheLoads=TheLoads
        self.Statuses=Statuses
        self.false_positive=false_positive
        self.false_negative=false_negative
        self.threshold=threshold

    def testresult(self,load):
        if bool(binomial(1, self.nonresprate)):
            return 'NonResponder'
        if self.threshold == None:
            if load in self.recognised:
                p = 1 - self.false_negative
            else:
                p = self.false_positive
        else:
            if load>self.threshold:
                p = 1 - self.false_negative
            else:
                p = self.false_positive
        if bool(binomial(1, p)):
            return 'Positive'
        else:
            return 'Negative'

    def __call__(self, sampling_times, people):
        if self.keeptrack:
            STATUSES = self.TheLoads.loc[sampling_times,self.sample()]
            return STATUSES.apply(self.testresults, )
        else:
            times_people=list(zip(sampling_times,people))
            raise Exception(str(times_people))
            STATUSES = list(map(lambda t:self.Statuses.loc[t[0],t[1]],times_people))
            return list(map(lambda x: x.apply(self.testresult),STATUSES))