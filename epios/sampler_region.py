from epios.sampler import Sampler
import pandas as pd
import numpy as np


class SamplerRegion(Sampler):
    '''
    The sampling class with region stratification.

    Parameters:
    ----------

    If you want to input new data, you can input that into data argument and set the pre_process to True
    If you want to use previous processed data, you can input the data_store_path to read data files,
    and set the pre_process to False.

    mode : str
        This indicates the specific mode to process the data.
        This should be the name of the modes that can be identified.

        **If you want this class sample as originally designed, do not change this value**

    '''

    def __init__(self, data=None, data_store_path='./input/', pre_process=True,
                 mode='Region'):
        self.mode = mode
        super().__init__(data=data, data_store_path=data_store_path,
                         pre_process=pre_process, mode=self.mode)
        geoinfo_path = data_store_path + 'microcells.csv'
        self.geoinfo = pd.read_csv(geoinfo_path)

    def get_region_dist(self):
        '''
        Extract the geo-distribution from the microcells.csv generated by DataProcess class

        Output:
        ------
        dist : list
            A list of floats, with sum 1, length should be the number of cells

        '''
        df = self.geoinfo
        n = df['Susceptible'].sum()
        dist = []
        for i in range(df['cell'].max() + 1):
            dist.append(df[df['cell'] == i]['Susceptible'].sum() / n)
        return dist

    def bool_exceed(self, current_region: int, cap_region: float):
        '''
        Return a boolean value to tell whether the sampling is going to exceed any cap
        --------
        Input:
        All inputs should be integers

        Output:
        True - means not reaching the cap
        False - means reaching the cap

        '''
        if current_region + 2 > cap_region:
            return False
        else:
            return True

    def multinomial_draw(self, n: int, prob: list):
        '''
        Perform a multinomial draw with caps, it will return a tuple of lists.
        The first output is the number of people that I want to draw from each group, specified by region.
        The second output is for convenience of the following sampling function.

        Parameters:
        ----------
        n : int
            The sample size
        prob : list
            List of floats, sum to 1. Length should be number of region groups

        Output:
        ------

        res : list
            A list of integers indicating the number of samples from each region group
        res_cap_block : list
            A list of caps for each region group

        '''
        # The following block trasform the probability to a list of barriers between 0 and 1
        # So we can use np.rand to generate a random number between 0 and 1 to
        # compare with the barriers to determine which group it is
        df = self.data
        prob = np.array(prob)
        if n > len(df):
            raise ValueError('Sample size should not be greater than population size')

        # Since we do not want too many samples from the same region group,
        # so we set a total cap for each region
        cap_region = []
        record_cap_region = []
        for i in range(len(prob)):
            cap_region.append(min(max(n * prob[i] + 0.005 * n, 1),
                                  self.geoinfo[self.geoinfo['cell'] == i]['Susceptible'].sum()))
            record_cap_region.append(self.geoinfo[self.geoinfo['cell'] == i]['Susceptible'].sum())
        cap_region = [cap_region, list(np.arange(len(cap_region)))]

        threshold = []
        for i in range(len(prob)):
            try:
                threshold.append(threshold[-1] + prob[i - 1])
            except IndexError:
                threshold.append(0)
        threshold.append(1)

        # Set the region counter to record whether any cap is reached
        res = [0] * len(prob)
        current_region = [0] * len(cap_region[0])

        # We start the draw from here, we run the following code for each sample
        # to determine which region group it is
        for i in range(n):
            rand = np.random.rand()
            j = 0
            while rand >= threshold[j]:
                j += 1
            # so the program will stop when it first exceed any barrier

            # Locate its position of region group
            j += -1
            pos_region = j

            # Use the above function to test whether it is going to hit the cap
            if self.bool_exceed(current_region[pos_region], cap_region[0][pos_region]):
                # This means it does not hit the cap
                res[int(cap_region[1][pos_region])] += 1
                current_region[pos_region] += 1
            else:
                # This means it hits the cap
                res[int(cap_region[1][pos_region])] += 1
                current_region[pos_region] += 1
                # Similar to the above
                prob_exceed = prob[pos_region]
                if i < n - 1:
                    if prob_exceed == prob.sum():
                        raise KeyError('Probability provided not supported for the sample size')
                prob = np.delete(prob, pos_region)
                if i < n - 1:
                    prob = prob / (1 - prob_exceed)
                cap_region = list(np.delete(np.array(cap_region), pos_region, 1))
                current_region.pop(pos_region)
                threshold = []
                for k in range(len(prob)):
                    try:
                        threshold.append(threshold[-1] + prob[k - 1])
                    except IndexError:
                        threshold.append(0)
                if len(threshold) > 0:
                    if threshold[-1] < 1:
                        threshold.append(1)
        return res, record_cap_region

    def sample(self, sample_size: int, additional_sample: list = None,
               household_criterion=False, household_threshold: int = 3):
        '''
        Given a sample size, and the additional sample, should return a list of people's IDs drawn from the population

        Parameters:
        ----------

        sample_size : int
            The size of sample
        additional_sample : list
            List of integers indicating the number of additional samples drawn from each region group
        household_criterion : bool
            Turn on or off the household criterion
        household_threshold : int
            The maximum number of people sampled from one household

        Output:
        ------

        res : list
            A list of ID of the sampled people

        '''
        res = []
        df = self.data

        # Get the region data
        region_dist = self.get_region_dist()

        # Assume region are two independent variables, calculate the prob
        # for a people in a specific region group
        ar_dist = np.array(region_dist)

        # We use the multinomial distribution to draw the samples, use the above
        # multinomial_draw function to achieve it
        size = sample_size
        num_sample, cap = self.multinomial_draw(size, ar_dist)
        num_sample = np.array(num_sample)

        # This is for non-responders, we may want some additional samples from due to this
        if additional_sample is None:
            pass
        else:
            additional_sample = np.array(additional_sample)
            num_sample = num_sample + additional_sample
            num_sample = np.array([min(elements) for elements in zip(cap, num_sample)])
        # Additional sample needs to be in the shape of num_region
        # With additional samples, need to be careful for post-processing

        # After we have the information of how many people we should draw from each region group,
        # Draw them using np.choice, which means completely at random
        # Then generate a list of IDs of the samples
        for i in range(len(num_sample)):
            ite = df[df['cell'] == i]
            ite_sample = list(ite['ID'])
            if household_criterion:
                count = 0
                while count < num_sample[i]:
                    if ite_sample:
                        pass
                    else:
                        raise ValueError('Household threshold is too low, not enough household to generate samples')
                    choice_ind = np.random.choice(np.arange(len(ite_sample)), size=1)[0]
                    choice = ite_sample[choice_ind]
                    if self.person_allowed(res, choice, threshold=household_threshold):
                        res.append(choice)
                        count += 1
                    ite_sample.pop(ite_sample.index(choice))
            else:
                choice = np.random.choice(np.arange(len(ite_sample)), size=num_sample[i], replace=False)
                for k in choice:
                    res.append(ite_sample[k])
        return res

    def additional_nonresponder(self, nonRespID: list, sampling_percentage=0.1, proportion=0.01, threshold=None):
        '''
        Generate the additional samples according to the non-responder IDs

        Parameters:
        ----------

        nonRespID : list
            A list containing the non-responder IDs
        sampling_percentage : float, between 0 and 1
            The proportion of additional samples taken from a specific regional group
        proportion : float, between 0 and 1
            The proportion of total groups to be sampled additionally
        threshold : NoneType or Int
            The lowest number of regional groups to be sampled additionally

        *Note: proportion and threshold both determine the number of groups to be sampled additionally,
               but both are depending on how many groups can be sampled additionally

        Output:
        ------

        additional_sample : list with length num_region_group
            A list containing how many additional samples we would like to draw from each region group

        '''
        num_region_group = len(self.get_region_dist())
        df = self.data
        n = num_region_group

        # Transform the nonRespID to nonRespNum to contain the number of non-responders
        # in each region group

        nonRespNum = [0] * (num_region_group)
        for i in nonRespID:
            region_group_pos = df[df['ID'] == i]['cell'].values[0]
            pos_nonRespRate = region_group_pos
            nonRespNum[pos_nonRespRate] += 1

        # Determine the number of groups to be sampled additionally
        if threshold is None:
            num_grp = round(proportion * n)
        else:
            num_grp = max(round(proportion * n), threshold)

        # Determine the position of groups to be resampled
        res = []
        for i in range(num_grp):
            if max(nonRespNum) > 0:
                pos = nonRespNum.index(max(nonRespNum))
                nonRespNum[i] = 0
                res.append(pos)

        # Determine the cap for each region groups
        additional_sample = [0] * num_region_group
        cap_region = []
        for i in range(len(nonRespNum)):
            pos_region = i
            ite = df[df['cell'] == pos_region]
            cap_region.append(len(ite))

        # Determine the number of additional samples from the above groups
        for i in res:
            additional_sample[i] = round(sampling_percentage * cap_region[i])
        return additional_sample
