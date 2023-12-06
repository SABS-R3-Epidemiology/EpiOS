from epios.data_process import DataProcess
import numpy as np
import pandas as pd


class Sampler(DataProcess):
    '''
    The base sampling class. This class will perform a totally random sampling.
    This class is inherited from the DataProcess class.

    '''

    def __init__(self, data=None, data_store_path='./input/', num_age_group=17):
        '''
        You can choose to import the data from a .csv file for a given path
        Or you can pass a pandas.DataFrame object to the 'data' argument,
        then we can use DataProcess class to process that data

        Warning: The DataFrame passed to DataProcess need to satisfy a certain format,
                 check DataProcess class for more details.

        '''
        if data is not None:
            super().__init__(data=data, path=data_store_path, num_age_group=num_age_group)
            self.data = pd.read_csv(data_store_path + 'data.csv')
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
        res = []
        sample = list(self.data['ID'])
        for i in choice:
            res.append(sample[i])
        return res

    def person_allowed(self, sample: list, choice: str, threshold: int = 3):
        """ function to see if the sampled person should be included in the generic sample

        Args:
            sample (list): list of people who have already been chosen
            choice (str): string id of the person being sampled
            threshold (int): the cap on the number of people sampled per household
        """

        # get the household of the person
        choice_household = '.'.join(choice.split('.')[:-1])

        # list of samples only showing first three numbers, e.g. "0.0.0" or "0.2.1"
        sample = ['.'.join(s.split('.')[:-1]) for s in sample]

        # get number of times that household is in sample list
        sample_count = sample.count(choice_household)

        # if adding this sample would exceed threshold then reject
        if sample_count >= threshold:

            return False

        # otherwise, return true
        else:

            return True
