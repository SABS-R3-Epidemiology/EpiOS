from epios.data_process import DataProcess
import numpy as np
import pandas as pd


class Sampler(DataProcess):
    '''
    The base sampling class.

    This class will perform a totally random sampling for a single time.

    Parameters:
    -----------

    If you want to input new data, you can input that into data argument and set the pre_process to True.

    If you want to use previous processed data, you can input the data_store_path to read data files,
    and set the pre_process to False.

    num_age_group : int
        This will be used when age stratification is enabled indicating how many age groups are there.

        *The last group includes age >= some threshold*
    age_group_width : int
        This will be used when age stratification is enabled,
        indicating the width of each age group (except for the last group)
    mode : str
        This indicates the specific mode to process the data.
        This should be the name of the modes that can be identified.

        **If you want this class sample as originally designed, do not change this value**

    '''

    def __init__(self, data=None, data_store_path='./input/', pre_process=True, num_age_group=None,
                 age_group_width=None, mode='Base'):
        self.mode = mode
        if data is not None:
            if pre_process:
                super().__init__(data=data, path=data_store_path, num_age_group=num_age_group,
                                 mode=self.mode, age_group_width=age_group_width)
            self.data = pd.read_csv(data_store_path + 'data.csv')
        else:
            if pre_process:
                raise ValueError('You have to input data to continue')
            else:
                self.data = pd.read_csv(data_store_path + 'data.csv')

    def sample(self, sample_size: int):
        '''
        This method samples data for a given sample size randomly.

        Parameters:
        -----------

        sample_size : int
            The size of sample

        Output:
        -------
        res : list
            A list of ID of people who is sampled

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
        """
        Function to see if the sampled person should be included in the generic sample

        Parameters:
        -----------

        sample : list
            List of people who have already been chosen
        choice : str
            string id of the person being sampled
        threshold : int
            The cap on the number of people sampled per household
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
