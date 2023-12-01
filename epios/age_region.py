import pandas as pd
import numpy as np
import json
import math
import mip


class Sampler():

    def __init__(self, geoinfo_path='./input/microcells.csv',
                 ageinfo_path='./input/pop_dist.json', data_path='./input/data.csv'):
        '''
        Contain all necessary information about the population
        ------------
        Input:
        geoinfo_path(str): should be the path to the csv file include the geo-information
        data(DataFrame): should be the extracted data to be sampled from the Epiabm

        '''
        self.geoinfo = pd.read_csv(geoinfo_path)
        self.ageinfo = ageinfo_path
        self.data = pd.read_csv(data_path)

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

    def bool_exceed(self, current_age: int, current_region: int,
                    current_block: int, cap_age: float, cap_region: float, cap_block: int):
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
        # So we can use np.rand to generate a random number between 0 and 1 to
        # compare with the barriers to determine which group it is
        df = self.data
        prob = np.array(prob)
        if n > len(df):
            raise ValueError('Sample size should not be greater than population size')

        # The following code generate the cap for each age-region group, since
        # there is a maximum the number of people in one age group in a region
        # The cap list will have shape (number of region, number of age groups)
        cap_block = []
        len_age = len(self.get_age_dist())
        record_age = len_age
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

        # Since we do not want too many samples from the same age/region group,
        # so we set a total cap for each age/region
        prob = prob.reshape((-1, len_age))
        cap_age = []
        cap_region = []
        for i in range(np.shape(prob)[1]):
            if i != np.shape(prob)[1] - 1:
                ite = df[df['age'] >= i * 5]
                ite = ite[ite['age'] < i * 5 + 5]
                max_num_age = len(ite)
                cap_age.append(min(n * prob[:, i].sum() + 0.01 * n, max_num_age))
            else:
                ite = df[df['age'] >= i * 5]
                max_num_age = len(ite)
                cap_age.append(min(max(n * prob[:, i].sum() + 0.01 * n, 1), max_num_age))
        cap_age = [cap_age, list(np.arange(len(cap_age)))]
        for i in range(np.shape(prob)[0]):
            cap_region.append(min(max(n * prob[i, :].sum() + 0.005 * n, 1),
                                  self.geoinfo[self.geoinfo['cell'] == i]['Susceptible'].sum()))
        cap_region = [cap_region, list(np.arange(len(cap_region)))]
        prob = prob.reshape((1, -1))[0]

        # Pre-process the prob, remove the prob for the age-region group with cap 0
        index_0 = []
        for i in range(len(cap_block)):
            if cap_block[i] == 0:
                index_0.append(i)
        for i in index_0:
            prob_exceed = prob[i]
            prob[i] = 0
            prob = prob / (1 - prob_exceed)
        threshold = []
        for i in range(len(prob)):
            try:
                threshold.append(threshold[-1] + prob[i - 1])
            except IndexError:
                threshold.append(0)
        threshold.append(1)
        cap_block = np.array(cap_block).reshape((-1, len_age))
        res_cap_block = cap_block.copy()

        # Set the age/region/block counter to record whether any cap is reached
        res = [0] * len(prob)
        current_age = [0] * len_age
        current_region = [0] * len(cap_region[0])
        current_block = np.zeros((len(cap_region[0]), len_age))

        # We start the draw from here, we run the following code for each sample
        # to determine which age/region group it is
        for i in range(n):
            rand = np.random.rand()
            for j in range(len(threshold)):
                if rand < threshold[j]:
                    # There is a break at the end of this if statement,
                    # so the program will stop when it first exceed any barrier

                    # Locate its position of age/region group
                    j += -1
                    pos_age = j % len_age
                    pos_region = math.floor(j / len_age)

                    # Use the above function to test whether it is going to hit the cap
                    if self.bool_exceed(current_age[pos_age], current_region[pos_region],
                                        current_block[pos_region, pos_age], cap_age[0][pos_age],
                                        cap_region[0][pos_region], cap_block[pos_region, pos_age]):
                        # This means it does not hit the cap
                        res[int(cap_region[1][pos_region] * record_age + cap_age[1][pos_age])] += 1
                        current_age[pos_age] += 1
                        current_region[pos_region] += 1
                        current_block[pos_region, pos_age] += 1
                    else:
                        # This means it hits the cap
                        res[int(cap_region[1][pos_region] * record_age + cap_age[1][pos_age])] += 1
                        current_age[pos_age] += 1
                        current_region[pos_region] += 1
                        current_block[pos_region, pos_age] += 1
                        prob = prob.reshape((-1, len_age))

                        # This is testing whether it hits block cap
                        if current_block[pos_region, pos_age] + 1 > cap_block[pos_region, pos_age]:
                            # reduce the corresponding prob to 0, and distribute its prob to the rest of blocks
                            prob_exceed = prob[pos_region, pos_age]
                            if i < len(df) - 1:
                                if prob_exceed == prob.sum():
                                    raise KeyError('Probability provided not supported for the sample size')
                            prob[pos_region, pos_age] = 0
                            prob = prob / (1 - prob_exceed)
                            prob = prob.reshape((1, -1))[0]
                            # Since the prob changes, need to calculate the threshold again
                            threshold = []
                            for k in range(len(prob)):
                                try:
                                    threshold.append(threshold[-1] + prob[k - 1])
                                except IndexError:
                                    threshold.append(0)
                            if threshold[-1] < 1:
                                threshold.append(1)
                            prob = prob.reshape((-1, len_age))

                        # Testing whether it hits age cap
                        if current_age[pos_age] + 1 > cap_age[0][pos_age]:
                            # Similarly, reduce all prob for this age group to 0, and re-distribute
                            prob_exceed = prob[:, pos_age].sum()
                            # if i < len(df) - 1:
                            #     if prob_exceed == prob.sum():
                            #         raise KeyError('Probability provided not supported for the sample size')
                            # These can be commented out since will be very unlikely to trigger
                            # Trigger this -> sum of prob of age group = 1 and reach the bound
                            # -> cannot be the bound in first arg of max above, since n * 1 + 0.01 * n > n
                            # -> the second arg means reach block cap for each age first
                            # -> dealt by the above if statement
                            # Note: problems may exist if prob.sum() < 1
                            # But, this error should be at least greater than 0.01 to influence this
                            # which is almost impossible
                            prob = np.delete(prob, pos_age, 1)
                            cap_block = np.delete(cap_block, pos_age, 1)
                            current_block = np.delete(current_block, pos_age, 1)
                            len_age += -1
                            prob = prob / (1 - prob_exceed)
                            prob = prob.reshape((1, -1))[0]
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
                            if len_age > 0:
                                prob = prob.reshape((-1, len_age))

                        # Testing whether it hits region cap
                        if current_region[pos_region] + 1 > cap_region[0][pos_region]:
                            # Similar to the above
                            prob_exceed = prob[pos_region, :].sum()
                            # if i < len(df) - 1:
                            #     if prob_exceed == prob.sum():
                            #         raise KeyError('Probability provided not supported for the sample size')
                            # See explaination above
                            prob = np.delete(prob, pos_region, 0)
                            cap_block = np.delete(cap_block, pos_region, 0)
                            current_block = np.delete(current_block, pos_region, 0)
                            prob = prob / (1 - prob_exceed)
                            prob = prob.reshape((1, -1))[0]
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
                            if len_age > 0:
                                prob = prob.reshape((-1, len_age))
                    break
        return res, res_cap_block

    def sample(self, sample_size: int, additional_sample: list = None):
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
        region_dist = self.get_region_dist()

        # Assume age and region are two independent variables, calculate the prob
        # for a people in a specific age-region group
        ar_dist = np.array(age_dist) * np.array(region_dist).reshape((-1, 1))
        ar_dist = ar_dist.reshape((1, -1))[0]

        # We use the multinomial distribution to draw the samples, use the above
        # multinomial_draw function to achieve it
        size = sample_size
        num_sample, cap = self.multinomial_draw(size, ar_dist)
        num_sample = np.array(num_sample)

        # This is for non-responders, we may want some additional samples from due to this
        if additional_sample is None:
            num_sample = num_sample.reshape((len(region_dist), -1))
        else:
            additional_sample = np.array(additional_sample)
            num_sample = num_sample.reshape((len(region_dist), -1)) + additional_sample
            num_sample = num_sample.reshape((1, -1))[0]
            cap = cap.reshape((1, -1))[0]
            num_sample = np.array([min(elements) for elements in zip(cap, num_sample)])
            num_sample = num_sample.reshape((len(region_dist), -1))
        # Additional sample needs to be in the shape of (num_region, num_age)
        # With additional samples, need to be careful for post-processing

        # After we have the information of how many people we should draw from each age-region group,
        # Draw them using np.choice, which means completely at random
        # Then generate a list of IDs of the samples
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
                for k in choice:
                    res.append(ite_sample[k])
        return res

    def optimise_draw(self):

        n = len(self.data)
        num_age = len(self.get_age_dist())
        num_region = len(self.get_region_dist())
        Q = np.zeros((num_age * num_region, num_age * num_region))
        for i in range(len(Q)):
            pos_age = i % num_age
            pos_region = math.floor(i / num_age)
            for j in range(num_age):
                Q[i, pos_region * num_age + j] = 1
            for j in range(num_region):
                Q[i, pos_age + j * num_region] = 1
        Q = list(Q)

        age_dist = self.get_age_dist()
        region_dist = self.get_region_dist()
        c = [0] * (num_age * num_region)
        for i in range(num_region * num_age):
            pos_age = i % num_age
            pos_region = math.floor(i / num_age)
            c[i] = -2 * n * (age_dist[pos_age] + region_dist[pos_region])

        cap_block = []
        num_age = len(self.get_age_dist())
        for i in range(num_age * num_region):
            pos_age = i % num_age
            pos_region = math.floor(i / num_age)
            ite = self.data[self.data['cell'] == pos_region]
            if pos_age != num_age - 1:
                ite = ite[ite['age'] >= pos_age * 5]
                ite = ite[ite['age'] < pos_age * 5 + 5]
            else:
                ite = ite[ite['age'] >= pos_age * 5]
            cap_block.append(len(ite))
        A1_ineq = list(np.eye(num_age * num_region))
        b1_ineq = cap_block

        cap_age = []
        cap_region = []
        for i in range(num_age):
            if i != num_age - 1:
                ite = self.data[self.data['age'] >= i * 5]
                ite = ite[ite['age'] < i * 5 + 5]
                max_num_age = len(ite)
                cap_age.append(min(n * age_dist[i] + 0.01 * n, max_num_age))
            else:
                ite = self.data[self.data['age'] >= i * 5]
                max_num_age = len(ite)
                cap_age.append(min(max(n * age_dist[i] + 0.01 * n, 1), max_num_age))
        cap_age = [cap_age, list(np.arange(len(cap_age)))]
        for i in range(num_region):
            cap_region.append(min(max(n * region_dist[i] + 0.005 * n, 1),
                                  self.geoinfo[self.geoinfo['cell'] == i]['Susceptible'].sum()))
        cap_region = [cap_region, list(np.arange(len(cap_region)))]
        A2_ineq = np.zeros((num_age, num_age * num_region))
        for i in range(num_age):
            for j in range(num_region):
                A2_ineq[i, i + j * num_age] = 1
        A2_ineq = list(A2_ineq)
        b2_ineq = cap_age

        A3_ineq = np.zeros((num_region, num_age * num_region))
        for i in range(num_region):
            for j in range(num_age):
                A3_ineq[i, i + j * num_region] = 1
        b3_ineq = cap_region

        A_eq = list(np.ones((1, num_age * num_region)))
        b_eq = [n]

        m = mip.Model()

        x = [m.add_var(var_type=mip.INTEGER) for i in range(len(c))]

        m.objective = mip.minimize(mip.xsum(Q[i][j] * x[i] * x[j] for i in range(len(x)) for j in range(len(x)))
                                   + mip.xsum(c[i] * x[i] for i in range(len(x))))

        for i in range(len(A1_ineq)):
            m += mip.xsum(A1_ineq[i][j] * x[j] for j in range(len(x))) <= b1_ineq[i]

        for i in range(len(A2_ineq)):
            m += mip.xsum(A2_ineq[i][j] * x[j] for j in range(len(x))) <= b2_ineq[i]

        for i in range(len(A3_ineq)):
            m += mip.xsum(A3_ineq[i][j] * x[j] for j in range(len(x))) <= b3_ineq[i]

        for i in range(len(A_eq)):
            m += mip.xsum(A_eq[i][j] * x[j] for j in range(len(x))) == b_eq[i]

        m.optimize()

        solution = [v.x for v in x]
        return solution
