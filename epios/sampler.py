from data_process import DataProcess
import numpy as np
import pandas as pd


class Sampler(DataProcess):
    '''
    The base sampling class. This class will perform a totally random sampling.
    This class is inherited from the DataProcess class.

    '''

    def __init__(self, data=None, data_store_path='./input/', num_age_group=17, data_path=None):
        '''
        You can choose to import the data from a .csv file for a given path
        Or you can pass a pandas.DataFrame object to the 'data' argument,
        then we can use DataProcess class to process that data

        Warning: The DataFrame passed to DataProcess need to satisfy a certain format,
                 check DataProcess class for more details.

        '''
        if (data_path is not None) and (data is not None):
            raise ValueError("You cannot input both 'data' and 'data_path'")
        elif data is not None:
            super().__init__(data=data, path=data_store_path, num_age_group=num_age_group)
            self.data = pd.read_csv(data_store_path + 'data.csv')
        elif data_path is not None:
            self.data = pd.read_csv(data_path)
        else:
            raise ValueError('You have to input data to continue')

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
