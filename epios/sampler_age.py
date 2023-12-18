from epios.sampler import Sampler
import numpy as np
import json


class SamplerAge(Sampler):

    def __init__(self, data=None, data_store_path='./input/', pre_process=True, num_age_group=17, age_group_width=5,
                 mode='Age'):
        '''
        Contain all necessary information about the population
        ------------
        Input:
        geoinfo_path(str): should be the path to the csv file include the geo-information
        data(DataFrame): should be the extracted data to be sampled from the Epiabm

        '''
        self.mode = mode
        super().__init__(data=data, data_store_path=data_store_path,
                         num_age_group=num_age_group, pre_process=pre_process,
                         age_group_width=age_group_width, mode=self.mode)
        ageinfo_path = data_store_path + 'pop_dist.json'
        self.ageinfo = ageinfo_path

    def get_age_dist(self):
        '''
        Read the age distribution from json file in EpiGeoPop
        ------------
        Input:
        path(str): should be the path to the file

        Output:
        config(list): should be a list of floats, with sum 1, length should be the number of age groups

        '''
        with open(self.ageinfo, 'r') as f:
            config = json.loads(f.read())
        return config

    def bool_exceed(self, current_age: int, cap_age: float):
        '''
        Return a boolean value to tell whether the sampling is going to exceed any cap
        --------
        Input:
        All inputs should be integers

        Output:
        True - means not reaching the cap
        False - means reaching the cap

        '''
        if current_age + 2 > cap_age:
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
        # So we can use np.rand to generate a random number between 0 and 1 to
        # compare with the barriers to determine which group it is
        df = self.data
        prob = np.array(prob)
        if n > len(df):
            raise ValueError('Sample size should not be greater than population size')

        len_age = len(self.get_age_dist())

        # Since we do not want too many samples from the same age/region group,
        # so we set a total cap for each age/region
        cap_age = []
        for i in range(len(prob)):
            if i != len(prob) - 1:
                ite = df[df['age'] >= i * self.age_group_width]
                ite = ite[ite['age'] < i * self.age_group_width + self.age_group_width]
                max_num_age = len(ite)
                cap_age.append(min(n * prob[i] + 0.01 * n, max_num_age))
            else:
                ite = df[df['age'] >= i * self.age_group_width]
                max_num_age = len(ite)
                cap_age.append(min(max(n * prob[i] + 0.01 * n, 1), max_num_age))
        cap_age = [cap_age, list(np.arange(len(cap_age)))]

        threshold = []
        for i in range(len(prob)):
            try:
                threshold.append(threshold[-1] + prob[i - 1])
            except IndexError:
                threshold.append(0)
        threshold.append(1)

        # Set the age/region/block counter to record whether any cap is reached
        res = [0] * len(prob)
        current_age = [0] * len_age

        # We start the draw from here, we run the following code for each sample
        # to determine which age/region group it is
        for i in range(n):
            rand = np.random.rand()
            j = 0
            while rand >= threshold[j]:
                j += 1
            # so the program will stop when it first exceed any barrier

            # Locate its position of age/region group
            j += -1
            pos_age = j

            # Use the above function to test whether it is going to hit the cap
            if self.bool_exceed(current_age[pos_age], cap_age[0][pos_age]):
                # This means it does not hit the cap
                res[int(cap_age[1][pos_age])] += 1
                current_age[pos_age] += 1
            else:
                # This means it hits the cap
                res[int(cap_age[1][pos_age])] += 1
                current_age[pos_age] += 1

                # Testing whether it hits age cap
                # Similarly, reduce all prob for this age group to 0, and re-distribute
                prob_exceed = prob[pos_age]
                if i < n - 1:
                    if prob_exceed == prob.sum():
                        raise KeyError('Probability provided not supported for the sample size')
                prob = np.delete(prob, pos_age)
                len_age += -1
                if i < n - 1:
                    prob = prob / (1 - prob_exceed)
                cap_age = list(np.delete(np.array(cap_age), pos_age, 1))
                current_age.pop(pos_age)
                threshold = []
                for k in range(len(prob)):
                    try:
                        threshold.append(threshold[-1] + prob[k - 1])
                    except IndexError:
                        threshold.append(0)
                if len(threshold) > 0:
                    if threshold[-1] < 1:
                        threshold.append(1)
        return res

    def sample(self, sample_size: int):
        '''
        Given a sample size, and the additional sample, should return a list of people's IDs drawn from the population
        ---------
        Input:
        sample_size(int): the size of sample
        additional_sample(list): list of integers indicating the number of additional
        samples drawn from each age-region group

        Output:
        res: a list of strings, each string is the ID of the sampled person

        '''
        res = []
        df = self.data

        # Get the age and region data
        age_dist = self.get_age_dist()

        # Assume age and region are two independent variables, calculate the prob
        # for a people in a specific age-region group
        ar_dist = np.array(age_dist)

        # We use the multinomial distribution to draw the samples, use the above
        # multinomial_draw function to achieve it
        size = sample_size
        num_sample = self.multinomial_draw(size, ar_dist)
        num_sample = np.array(num_sample)

        # After we have the information of how many people we should draw from each age-region group,
        # Draw them using np.choice, which means completely at random
        # Then generate a list of IDs of the samples
        for j in range(len(num_sample)):
            if j != len(num_sample) - 1:
                ite = df[df['age'] >= j * self.age_group_width]
                ite = ite[ite['age'] < j * self.age_group_width + self.age_group_width]
            else:
                ite = df[df['age'] >= j * self.age_group_width]
            ite_sample = list(ite['ID'])
            choice = np.random.choice(np.arange(len(ite_sample)), size=num_sample[j], replace=False)
            for k in choice:
                res.append(ite_sample[k])
        return res
