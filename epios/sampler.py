from data_process import DataProcess
import numpy as np
import pandas as pd


class Sampler(DataProcess):
    '''
    The base sampling class. This class will perform a totally random sampling.
    This class is inherited from the DataProcess class.

    '''

    def __init__(self, data_path='./input/data.csv', data=None):
        '''
        You can choose to import the data from a .csv file for a given path
        Or you can pass a pandas.DataFrame object to the 'data' argument,
        then we can use DataProcess class to process that data

        Warning: The DataFrame passed to DataProcess need to satisfy a certain format,
                 check DataProcess class for more details.

        '''
        if data is not None:
            super().__init__(data)
        self.data = pd.read_csv(data_path)

    def sample(self, sample_size: int):
        '''
        This method samples data for a given sample size randomly.
        -------
        Input:
        sample_size(int): The size of sample

        Output:
        res(list): a list of ID of people who is sampled

        '''
        population_size = len(self.data)
        if sample_size > population_size:
            raise ValueError('Sample size should <= population size')
        choice = np.random.choice(np.arange(population_size), size=sample_size, replace=False)
        res = list(self.data['ID'])[choice]
        return res
