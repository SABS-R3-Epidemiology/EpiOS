from epios.sampler import Sampler
import pandas as pd
import numpy as np
import json
import math
# from gurobipy import Model, GRB, quicksum


class SamplerAgeRegion(Sampler):

    def __init__(self, data=None, data_store_path='./input/', num_age_group=17, geoinfo_path='./input/microcells.csv',
                 ageinfo_path='./input/pop_dist.json', data_path=None):
        '''
        Contain all necessary information about the population
        ------------
        Input:
        geoinfo_path(str): should be the path to the csv file include the geo-information
        data(DataFrame): should be the extracted data to be sampled from the Epiabm

        '''
        super().__init__(data_path=data_path, data=data, data_store_path=data_store_path,
                         num_age_group=num_age_group)
        self.geoinfo = pd.read_csv(geoinfo_path)
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

    def sample(self, sample_size: int, additional_sample: list = None,
               household_criterion=False, household_threshold: int = 3):
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
                if household_criterion:
                    count = 0
                    while count < num_sample[i, j]:
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
                    choice = np.random.choice(np.arange(len(ite_sample)), size=num_sample[i, j], replace=False)
                    for k in choice:
                        res.append(ite_sample[k])
        return res

    def additional_nonresponder(self, nonRespID: list, sampling_percentage=0.1, proportion=0.01, threshold=None):
        '''
        Generate the additional samples according to the non-response rate
        --------
        Input:
        nonRespID(list): A list containing the non-responder IDs
        sampling_percentage(float, between 0 and 1): The proportion of additional samples
                                                        taken from a specific age-regional group
        proportion(float, between 0 and 1): The proportion of total groups to be sampled additionally
        threshold(NoneType or Int): The lowest number of age-regional groups to be sampled additionally

        Note: proportion and threshold both determined the number of groups to be sampled additionally,
                But both are depending on how many groups can be sampled additionally

        Output:
        additional_sample(list of 2D, with dimension (num_region_group, num_age_group)):
                        A list containing how many additional samples we would like to draw
                        from each age-region group

        '''
        num_age_group = len(self.get_age_dist())
        num_region_group = len(self.get_region_dist())
        df = self.data
        n = num_age_group * num_region_group

        # Transform the nonRespID to nonRespNum to contain the number of non-responders
        # in each age-region group

        nonRespNum = [0] * (num_age_group * num_region_group)
        for i in nonRespID:
            age = df[df['ID'] == i]['age'].values[0]
            if math.floor(age / 5) < num_age_group - 1:
                age_group_pos = math.floor(age / 5)
            else:
                age_group_pos = num_age_group - 1
            region_group_pos = df[df['ID'] == i]['cell'].values[0]
            pos_nonRespRate = region_group_pos * num_age_group + age_group_pos
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
                res.append([pos % num_age_group, math.floor(pos / num_age_group)])

        # Determine the cap for each age-region groups
        additional_sample = list(np.zeros((num_region_group, num_age_group)))
        cap_block = []
        for i in range(len(nonRespNum)):
            pos_age = i % num_age_group
            pos_region = math.floor(i / num_age_group)
            ite = df[df['cell'] == pos_region]
            if pos_age != num_age_group - 1:
                ite = ite[ite['age'] >= pos_age * 5]
                ite = ite[ite['age'] < pos_age * 5 + 5]
            else:
                ite = ite[ite['age'] >= pos_age * 5]
            cap_block.append(len(ite))
        cap_block = np.array(cap_block).reshape((-1, num_age_group))

        # Determine the number of additional samples from the above groups
        for i in res:
            additional_sample[i[1]][i[0]] = round(sampling_percentage * cap_block[i[1], i[0]])
        return additional_sample

    # def optimise_draw(self, sample_size: int):
    #     '''
    #     This function use package gurobipy to solve a MIQP problem to replace the multinomial_draw
    #     function above to generate the optimal number of people drawn from each age-region group
    #     ----------
    #     Prerequisites:
    #     gurobipy package and an academic license required!
    #     To install this package, you need to firstly register a free academic account on Guroby
    #     website, and submit a request for a free license
    #     Secondly, you can pip install gurobipy, then run the command provided when applying the license
    #     to install that license. Then you finish the setup!

    #     Input:
    #     sample_size(int): the size of sample

    #     Output:
    #     res(list): If an optimal solution is found:
    #                An 1D list of numbers, with length number of age groups * number of region groups,
    #                indicating the number of people should be drawn from each age-region group
    #                Otherwise:
    #                Raise an ValueError, please go and check your sample size!

    #     '''
    #     # Setup the matrix Q in the MIQP problem
    #     num_age = len(self.get_age_dist())
    #     num_region = len(self.get_region_dist())
    #     Q = np.zeros((num_age * num_region, num_age * num_region))
    #     for i in range(len(Q)):
    #         pos_age = i % num_age
    #         pos_region = math.floor(i / num_age)
    #         for j in range(num_age):
    #             Q[i, pos_region * num_age + j] = 1
    #         for j in range(num_region):
    #             Q[i, pos_age + j * num_age] = 1
    #     Q = list(Q)

    #     # Setup the vector c in the MIQP problem
    #     age_dist = self.get_age_dist()
    #     region_dist = self.get_region_dist()
    #     c = [0] * (num_age * num_region)
    #     for i in range(num_region * num_age):
    #         pos_age = i % num_age
    #         pos_region = math.floor(i / num_age)
    #         c[i] = -2 * sample_size * (age_dist[pos_age] + region_dist[pos_region])

    #     # Setup the constraint to keep the number picked for each age-region group
    #     # does not exceed the existed number of people within that group
    #     cap_block = []
    #     num_age = len(self.get_age_dist())
    #     for i in range(num_age * num_region):
    #         pos_age = i % num_age
    #         pos_region = math.floor(i / num_age)
    #         ite = self.data[self.data['cell'] == pos_region]
    #         if pos_age != num_age - 1:
    #             ite = ite[ite['age'] >= pos_age * 5]
    #             ite = ite[ite['age'] < pos_age * 5 + 5]
    #         else:
    #             ite = ite[ite['age'] >= pos_age * 5]
    #         cap_block.append(len(ite))
    #     A1_ineq = list(np.eye(num_age * num_region))
    #     b1_ineq = cap_block

    #     # Setup the constraint for cap of age groups
    #     cap_age = []
    #     cap_region = []
    #     for i in range(num_age):
    #         if i != num_age - 1:
    #             ite = self.data[self.data['age'] >= i * 5]
    #             ite = ite[ite['age'] < i * 5 + 5]
    #             max_num_age = len(ite)
    #             cap_age.append(min(sample_size * age_dist[i] + 0.01 * sample_size, max_num_age))
    #         else:
    #             ite = self.data[self.data['age'] >= i * 5]
    #             max_num_age = len(ite)
    #             cap_age.append(min(max(sample_size * age_dist[i] + 0.01 * sample_size, 1), max_num_age))
    #     for i in range(num_region):
    #         cap_region.append(min(max(sample_size * region_dist[i] + 0.005 * sample_size, 1),
    #                               self.geoinfo[self.geoinfo['cell'] == i]['Susceptible'].sum()))
    #     A2_ineq = np.zeros((num_age, num_age * num_region))
    #     for i in range(num_age):
    #         for j in range(num_region):
    #             A2_ineq[i, i + j * num_age] = 1
    #     A2_ineq = list(A2_ineq)
    #     b2_ineq = cap_age

    #     # Setup the constraint for caps of region groups
    #     A3_ineq = np.zeros((num_region, num_age * num_region))
    #     for i in range(num_region):
    #         for j in range(num_age):
    #             A3_ineq[i, i * num_age + j] = 1
    #     b3_ineq = cap_region

    #     # Make sure that the sum of number of people sampled equals to the sample size
    #     A_eq = list(np.ones((1, num_age * num_region)))
    #     b_eq = [sample_size]

    #     len_c = len(c)

    #     # Construct the model
    #     m = Model('miqp')

    #     x = m.addVars(len(Q), vtype=GRB.INTEGER)

    #     obj = quicksum(quicksum(Q[i][j] * x[i] * x[j] for j in range(len_c)) for i in range(len_c))
    #     obj += quicksum(c[i] * x[i] for i in range(len_c))
    #     m.setObjective(obj, GRB.MINIMIZE)

    #     # Add the constraints defined above
    #     for i in range(len(A1_ineq)):
    #         m.addConstr(quicksum(A1_ineq[i][j] * x[j] for j in range(len_c)) <= b1_ineq[i], "constraint{}".format(i))

    #     for i in range(len(A2_ineq)):
    #         m.addConstr(quicksum(A2_ineq[i][j] * x[j] for j in range(len_c)) <= b2_ineq[i], "constraint{}".format(i))

    #     for i in range(len(A3_ineq)):
    #         m.addConstr(quicksum(A3_ineq[i][j] * x[j] for j in range(len_c)) <= b3_ineq[i], "constraint{}".format(i))

    #     for i in range(len(A_eq)):
    #         m.addConstr(quicksum(A_eq[i][j] * x[j] for j in range(len_c)) == b_eq[i], "constraint{}".format(i))

    #     m.optimize()

    #     # Return the result
    #     if m.status == GRB.Status.OPTIMAL:
    #         res = [i.x for i in m.getVars()]
    #         return res
    #     else:
    #         raise ValueError('No solution exists for these constraints, check sample_size please')
