import pandas as pd
import numpy as np
import json
import math


class Sampler():

    def __init__(self, geoinfo_path: str, data):
        '''
        Contain all necessary information about the population
        ------------
        Input:
        geoinfo_path(str): should be the path to the csv file include the geo-information
        data(DataFrame): should be the extracted data to be sampled from the Epiabm
        
        '''
        self.geoinfo = pd.read_csv(geoinfo_path)
        self.data = data

    def get_age_dist(self, path: str):
        '''
        Read the age distribution from json file in EpiGeoPop
        ------------
        Input:
        path(str): should be the path to the file

        Output:
        config(list): should be a list of floats, with sum 1, length should be the number of age groups

        '''
        with open(path, 'r') as f:
            cofig = json.loads(f.read())
        return cofig

    def get_region_dist(self):
        '''
        Extract the geo-distribution from the geo-information
        ----------
        Output:
        dist(list): should be a list of floats, with sum 1, length should be the number of cells

        '''
        df = self.geoinfo
        n = df['Susceptible'].sum()
        dist = []
        for i in range(df['cell'].max() + 1):
            dist.append(df[df['cell'] == i]['Susceptible'].sum() / n)
        return dist

    def bool_exceed(self, current_age: int, current_region: int, current_block: int, cap_age: int, cap_region: int, cap_block: int):
        '''
        Return a boolean value to tell whether the sampling is going to exceed any cap
        --------
        Input:
        All inputs should be integers

        Output:
        True - means not reaching the cap
        False - means reaching the cap

        '''
        if current_block + 2 > cap_block:
            return False
        elif current_age + 2 > cap_age:
            return False
        elif current_region + 2 > cap_region:
            return False
        else:
            return True

    def multinomial_draw(self, n: int, prob: list):
        '''
        Perform a multinomial draw with caps, it will return a tuple of lists
        The first output is the number of people that I want to draw from each group, specified by age and region
        The second output is for convenience of the following sampling function
        ---------
        Input:
        n(int): the sample size
        prob(list): list of floats, sum to 1, length should be number of age groups times number of region groups

        Output:
        res(list): a list of integers indicating the number of samples from each age-region group
        res_cap_block(list): a list of caps for each age-region group

        '''
        # The following block trasform the probability to a list of barriers between 0 and 1
        # So we can use np.rand to generate a random number between 0 and 1 to compare with the barriers to determine which group it is
        df = self.data
        threshold = []
        for i in len(prob):
            try:
                threshold.append(threshold[-1] + prob[i - 1])
            except:
                threshold.append(0)

        # The following code generate the cap for each age-region group, since there is a maximum the number of people in one age group in a region
        # The cap list will have shape (number of region, number of age groups)
        cap_block = []
        len_age = len(self.get_age_dist())
        for i in range(len(prob)):
            pos_age = i % len_age
            pos_region = math.floor(i / len_age)
            ite = df[df['cell'] == pos_region]
            if pos_age != len_age - 1:
                ite = ite[ite['age'] >= pos_age * 5]
                ite = ite[ite['age'] < pos_age * 5 + 5]
            else:
                ite = ite[ite['age'] >= pos_age * 5]
            cap_block.append(len(ite))
        cap_block = cap_block.reshape((-1, len_age))
        res_cap_block = cap_block.copy()

        # Since we do not want too many samples from the same age/region group, so we set a total cap for each age/region
        prob = prob.reshape((-1, len_age))
        cap_age = []
        cap_region = []
        for i in range(np.shape(prob)[1]):
            cap_age.append(n * prob[:, i].sum() + 0.01 * n)
        for i in range(np.shape(prob)[0]):
            cap_region.append(min(n * prob[i, :].sum() + 0.005 * n, self.geoinfo[self.geoinfo['cell'] == i]['Susceptible'].sum()))
        prob = prob.reshape((1, -1))

        # Set the age/region/block counter to record whether any cap is reached
        res = [0] * len(prob)
        current_age = [0] * len_age
        current_region = [0] * len(cap_region)
        current_block = [[0] * len_age] * len(cap_region)

        # We start the draw from here, we run the following code for each sample to determine which age/region group it is
        for i in range(n):
            rand = np.random.rand()
            for j in range(len(threshold)):
                if rand > threshold[j]: # There is a break at the end of this if statement, so the program will 
                    pos_age = j % len_age
                    pos_region = math.floor(j / len_age)
                    if self.bool_exceed(current_age[pos_age], current_region[pos_region], current_block[pos_region, pos_age], cap_age[pos_age], cap_region[pos_region], cap_block[pos_region, pos_age]):
                        res[j] += 1
                        current_age[pos_age] += 1
                        current_region[pos_region] += 1
                        current_block[pos_age, pos_region] += 1
                    else:
                        res[j] += 1
                        current_age[pos_age] += 1
                        current_region[pos_region] += 1
                        prob = prob.reshape((-1, len_age))
                        if current_block[pos_region, pos_age] + 1 > cap_block[pos_region, pos_age]:
                            prob_exceed = prob[pos_region, pos_age]
                            prob[pos_region, pos_age] = 0
                            prob = prob / (1 - prob_exceed)
                            prob = prob.reshape((1, -1))
                            threshold = []
                            for k in len(prob):
                                try:
                                    threshold.append(threshold[-1] + prob[i - 1])
                                except:
                                    threshold.append(0)
                            prob = prob.reshape((-1, len_age))
                        if current_age[pos_age] + 1 > cap_age[pos_age]:
                            prob_exceed = prob[:, pos_age].sum()
                            np.delete(prob, pos_age, 1)
                            np.delete(cap_block, pos_age, 1)
                            prob = prob / (1 - prob_exceed)
                            prob = prob.reshape((1, -1))
                            cap_age.pop(pos_age)
                            threshold = []
                            for k in len(prob):
                                try:
                                    threshold.append(threshold[-1] + prob[i - 1])
                                except:
                                    threshold.append(0)
                            prob = prob.reshape((-1, len_age))
                        if current_region[pos_region] + 1 > cap_region[pos_region]:
                            prob_exceed = prob[pos_region, :].sum()
                            np.delete(prob, pos_region, 0)
                            np.delete(cap_block, pos_region, 0)
                            prob = prob / (1 - prob_exceed)
                            prob = prob.reshape((1, -1))
                            cap_region.pop(pos_region)
                            threshold = []
                            for k in len(prob):
                                try:
                                    threshold.append(threshold[-1] + prob[i - 1])
                                except:
                                    threshold.append(0)
                            prob = prob.reshape((-1, len_age))
                    break
        return res, res_cap_block

    def sample(self, sample_size: int, additional_sample: list=None):
        '''
        Given a sample size, and the additional sample, should return a list of people's IDs drawn from the population
        ---------
        Input:
        sample_size(int): the size of sample
        additional_sample(list): list of integers indicating the number of additional samples drawn from each age-region group

        Output:
        res: a list of strings, each string is the ID of the sampled person

        '''
        res = []
        df = self.data
        age_dist = self.get_age_dist()
        region_dist = self.get_region_dist()
        ar_dist = np.array(age_dist) * np.array(region_dist).reshape((-1, 1))
        ar_dist = ar_dist.reshape((1, -1))
        size = sample_size
        num_sample, cap = self.multinomial_draw(size, ar_dist)
        if additional_sample is None:
            num_sample = num_sample.reshape((len(region_dist), -1))
        else:
            num_sample = num_sample.reshape((len(region_dist), -1)) + additional_sample
            num_sample = np.array([min(elements) for elements in zip(cap, num_sample)])
        # Additional sample needs to be in the shape of (num_region, num_age)
        # With additional samples, need to be careful for post-processing
        for i in range(len(num_sample)):
            for j in range(len(num_sample[0])):
                if j != len(num_sample[0]) - 1:
                    ite = df[df['cell'] == i]
                    ite = ite[ite['age'] >= j * 5]
                    ite = ite[ite['age'] < j * 5 + 5]
                else:
                    ite = df[df['cell'] == i]
                    ite = ite[ite['age'] >= j * 5]
                ite_sample = list(ite['ID'])
                choice = np.random.choice(np.arange(len(ite_sample)), size=num_sample[i, j], replace=False)
                res.append(ite_sample[choice])
        return res
