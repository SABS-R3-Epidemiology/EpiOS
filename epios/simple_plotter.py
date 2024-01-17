from numpy import sum
from re_scaler import ReScaler
from pandas import DataFrame
from matplotlib import pyplot

class SimplePlotter():


    '''
    Class to plot the information returned
    
    '''


    def __init__(self, false_positive=0, false_negative=0):
        self.false_positive = false_positive
        self.false_negative = false_negative


    def __call__(self, observation):
        if type(observation)==DataFrame:
            observed=observation.apply(sum,observation)
            times = observed.index
            observed = observed.values
        else:
            observed = list(map(lambda x: x.apply(sum).values[0],observation))
            times = list(map(lambda x: x.index[0],observation))
        observed = ReScaler(false_positive = self.false_positive, false_negative = self.false_negative)(observed)
        pyplot.plot(times,observed)