import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from itertools import product
import multiprocessing
import sys
import os
from epios import Sampler
from epios import SamplerAgeRegion
from epios import SamplerAge
from epios import SamplerRegion
from epios import SamplingMaker


class PostProcess():
    '''
    This class is to automatically sample the population at several given time points.

    And generate plots and comparison with the true infection level within the population.

    How to use:
    -----------

    Define an instance and input the demographical and time data of the population
    Then use __call__ method directly to generate plots and comparison

    To define an instance of PostProcess, you need the following inputs:

    Parameters:
    -----------

    demo_data : pandas.DataFrame
        The geographical data of the population
    time_data : pandas.DataFrame
        The infection data of the population at different time points

    '''

    def __init__(self, demo_data: pd.DataFrame, time_data: pd.DataFrame):
        self.demo_data = demo_data
        self.time_data = time_data

    def __call__(self, sampling_method, sample_size, time_sample, non_responder=False, comparison=True,
                 non_resp_rate=None, data_store_path='./input/', **kwargs):
        '''
        Make this class callable to run the post process part just by calling the instance

        Parameters:
        -----------

        sampling_method : str
            A specific string tells which sampling method using

            Methods can be recognised: AgeRegion, Region, Age, Base
        sample_size : int
            The size of sample
        time_sample : list
            A list of time points to sample the population
        non_responder : bool
            Turn on or off the non-responder function
        comparison : bool
            Turn on or off the comparison between the sampled result and the true result
        kwargs : dict
            A dictionary of parameters passed to process part
            The following parameters can be passed:
                gen_plot : bool
                    Whether or not to generate plots

                    Default = False
                saving_path_sampling : str
                    Path to save figures of sampled data

                    Default = None
                saving_path_compare : str
                    Path to save figures of comparison data

                    Default = None
                num_age_group : int
                    Indicating how many age groups are there.

                    *The last group includes age >= some threshold*

                    Default = 17
                age_group_width : int
                    Indicating the width of each age group(except for the last group)

                    Default = 5
                scale_method : str
                    Specific string telling how to compare the sampled data with the true population

                    Default = 'proportional'
                sampling_percentage : float, between 0 and 1
                    The proportion of additional samples taken from a specific (age-)regional group

                    Default = 0.1 (Only for non-responders)
                proportion : float, between 0 and 1
                    The proportion of total groups to be sampled additionally

                    Default = 0.01 (Only for non-responders)
                threshold : NoneType or Int
                    The lowest number of groups to be sampled additionally

                    Default = None (Only for non-responders)
                data_store_path : str
                    The path to store data generated during sampling

                    Default = ./input/

        '''
        if non_responder:  # For non-responders enabled
            if non_resp_rate is None:
                raise ValueError('You have to input the non-response rate when considering non-responders')

            # Select all useful variable names provided in kwargs
            sampling_params = ['gen_plot', 'saving_path_sampling', 'num_age_group', 'age_group_width',
                               'sampling_percentage', 'proportion', 'threshold']
            compare_params = ['scale_method', 'saving_path_compare', 'gen_plot']
            total_params = set(sampling_params + compare_params)

            # Print the unused variable names
            params_not_used = [param for param in kwargs if param not in total_params]
            if params_not_used:
                print_str = 'The following parameters provided are not used: '
                print_str += ', '.join(params_not_used)
                print(print_str)

            # Pass the parameters for sampling into the function
            sampling_input = {}
            for i in sampling_params:
                try:
                    sampling_input[i] = kwargs[i]
                except KeyError:
                    pass
            res = self._sampled_non_responder(sampling_method=sampling_method, sample_size=sample_size,
                                              time_sample=time_sample, non_resp_rate=non_resp_rate,
                                              data_store_path=data_store_path, **sampling_input)
            # Pass the parameters for comparison into the function
            if comparison:
                compare_input = {}
                for i in compare_params:
                    try:
                        compare_input[i] = kwargs[i]
                    except KeyError:
                        pass
                diff = self._compare(time_sample=time_sample, **compare_input)
                return res, diff
            return res, None
        else:  # For non-responders disabled
            # Almost the same script, see comments above
            sampling_params = ['sample_strategy', 'gen_plot', 'saving_path_sampling', 'num_age_group',
                               'age_group_width']
            compare_params = ['scale_method', 'saving_path_compare', 'gen_plot']
            total_params = set(sampling_params + compare_params)
            params_not_used = []
            for i in kwargs:
                if i in total_params:
                    pass
                else:
                    params_not_used.append(i)
            if params_not_used:
                print_str = 'The following parameters provided are not used: '
                for i in params_not_used:
                    print_str += i
                    print_str += ', '
                print_str = print_str[:-2]
                print(print_str)
            sampling_input = {}
            for i in sampling_params:
                try:
                    sampling_input[i] = kwargs[i]
                except KeyError:
                    pass
            res = self._sampled_result(sampling_method=sampling_method, sample_size=sample_size,
                                       time_sample=time_sample, data_store_path=data_store_path,
                                       **sampling_input)
            if comparison:
                compare_input = {}
                for i in compare_params:
                    try:
                        compare_input[i] = kwargs[i]
                    except KeyError:
                        pass
                diff = self._compare(time_sample=time_sample, **compare_input)
                return res, diff
            return res, None

    def _sampled_result(self, sampling_method, sample_size, time_sample, sample_strategy='Random',
                        gen_plot: bool = False, saving_path_sampling=None, num_age_group=17, age_group_width=5,
                        data_store_path='./input/'):
        '''
        This is a method to generate the sampled result and plot a figure
        This method should not be used directly, it is integrated within the __callable__ method

        '''
        if sampling_method == 'AgeRegion':  # For both age and region stratification
            if sample_strategy == 'Same':  # Do not change people sampled at each sample time point
                infected_rate = []

                # Do the sampling
                sampler_class = SamplerAgeRegion(data=self.demo_data, data_store_path=data_store_path,
                                                 num_age_group=num_age_group, age_group_width=age_group_width)
                people = sampler_class.sample(sample_size=sample_size)

                # Get results of each people sampled
                X = SamplingMaker(non_resp_rate=0, keeptrack=True, TheData=self.time_data,
                                  false_positive=0, false_negative=0, threshold=None)
                ite = X(time_sample, people)

                # Output the infected rate
                for i in range(len(time_sample)):
                    infected_rate.append(ite.iloc[i].value_counts().get('Positive', 0) / len(people))
            elif sample_strategy == 'Random':  # Change people sampled at each sample time point
                infected_rate = []
                for i in range(len(time_sample)):  # Sample at each sample time points
                    if i == 0:  # First time sampling, need pre_process
                        sampler_class = SamplerAgeRegion(data=self.demo_data, data_store_path=data_store_path,
                                                         num_age_group=num_age_group, age_group_width=age_group_width)
                    else:  # After the data process, we can directly read files processed at the first time
                        sampler_class = SamplerAgeRegion(data_store_path=data_store_path, pre_process=False,
                                                         num_age_group=num_age_group, age_group_width=age_group_width)
                    people = sampler_class.sample(sample_size=sample_size)

                    # Get the results of each people sampled
                    X = SamplingMaker(non_resp_rate=0, keeptrack=True, TheData=self.time_data,
                                      false_positive=0, false_negative=0, threshold=None)
                    ite = X([time_sample[i]], people)

                    # Output the infected rate
                    infected_rate.append(ite.iloc[0].value_counts().get('Positive', 0) / len(people))
        elif sampling_method == 'Age':  # For only age stratification
            # Almost the same code as above, see comments above
            if sample_strategy == 'Same':
                infected_rate = []
                sampler_class = SamplerAge(data=self.demo_data, data_store_path=data_store_path,
                                           num_age_group=num_age_group, age_group_width=age_group_width)
                people = sampler_class.sample(sample_size=sample_size)
                X = SamplingMaker(non_resp_rate=0, keeptrack=True, TheData=self.time_data,
                                  false_positive=0, false_negative=0, threshold=None)
                ite = X(time_sample, people)
                for i in range(len(time_sample)):
                    infected_rate.append(ite.iloc[i].value_counts().get('Positive', 0) / len(people))
            elif sample_strategy == 'Random':
                infected_rate = []
                for i in range(len(time_sample)):
                    if i == 0:
                        sampler_class = SamplerAge(data=self.demo_data, data_store_path=data_store_path,
                                                   num_age_group=num_age_group, age_group_width=age_group_width)
                    else:
                        sampler_class = SamplerAge(data_store_path=data_store_path, pre_process=False,
                                                   num_age_group=num_age_group, age_group_width=age_group_width)
                    people = sampler_class.sample(sample_size=sample_size)
                    X = SamplingMaker(non_resp_rate=0, keeptrack=True, TheData=self.time_data,
                                      false_positive=0, false_negative=0, threshold=None)
                    ite = X([time_sample[i]], people)
                    infected_rate.append(ite.iloc[0].value_counts().get('Positive', 0) / len(people))
        elif sampling_method == 'Region':  # Only region stratification
            # Almost same code as above, see comments above
            if sample_strategy == 'Same':
                infected_rate = []
                sampler_class = SamplerRegion(data=self.demo_data, data_store_path=data_store_path)
                people = sampler_class.sample(sample_size=sample_size)
                X = SamplingMaker(non_resp_rate=0, keeptrack=True, TheData=self.time_data,
                                  false_positive=0, false_negative=0, threshold=None)
                ite = X(time_sample, people)
                for i in range(len(time_sample)):
                    infected_rate.append(ite.iloc[i].value_counts().get('Positive', 0) / len(people))
            elif sample_strategy == 'Random':
                infected_rate = []
                for i in range(len(time_sample)):
                    if i == 0:
                        sampler_class = SamplerRegion(data=self.demo_data, data_store_path=data_store_path)
                    else:
                        sampler_class = SamplerRegion(data_store_path=data_store_path, pre_process=False)
                    people = sampler_class.sample(sample_size=sample_size)
                    X = SamplingMaker(non_resp_rate=0, keeptrack=True, TheData=self.time_data,
                                      false_positive=0, false_negative=0, threshold=None)
                    ite = X([time_sample[i]], people)
                    infected_rate.append(ite.iloc[0].value_counts().get('Positive', 0) / len(people))
        elif sampling_method == 'Base':  # Use the base sampling method, no age/regional stratification
            # Almost the same code as above, see comments above
            if sample_strategy == 'Same':
                infected_rate = []
                sampler_class = Sampler(data=self.demo_data, data_store_path=data_store_path)
                people = sampler_class.sample(sample_size=sample_size)
                X = SamplingMaker(non_resp_rate=0, keeptrack=True, TheData=self.time_data,
                                  false_positive=0, false_negative=0, threshold=None)
                ite = X(time_sample, people)
                for i in range(len(time_sample)):
                    infected_rate.append(ite.iloc[i].value_counts().get('Positive', 0) / len(people))
            elif sample_strategy == 'Random':
                infected_rate = []
                for i in range(len(time_sample)):
                    if i == 0:
                        sampler_class = Sampler(data=self.demo_data, data_store_path=data_store_path)
                    else:
                        sampler_class = Sampler(data_store_path=data_store_path, pre_process=False)
                    people = sampler_class.sample(sample_size=sample_size)
                    X = SamplingMaker(non_resp_rate=0, keeptrack=True, TheData=self.time_data,
                                      false_positive=0, false_negative=0, threshold=None)
                    ite = X([time_sample[i]], people)
                    infected_rate.append(ite.iloc[0].value_counts().get('Positive', 0) / len(people))
        else:
            raise ValueError('You must input a valid sampling method')

        # Plot the figure
        if gen_plot:
            plt.plot(time_sample, infected_rate)
            plt.xlabel('Time')
            plt.ylabel('Population')
            plt.xlim(0, max(time_sample))
            plt.ylim(0, len(self.demo_data))
            plt.title('Number of infection in the sample')
            if saving_path_sampling:
                plt.savefig(saving_path_sampling)
        res = []
        res.append(time_sample)
        res.append(infected_rate)
        # Output the results for comparison use
        self.result = infected_rate
        return res

    def _sampled_non_responder(self, sampling_method, sample_size, time_sample, non_resp_rate,
                               gen_plot: bool = False, saving_path_sampling=None, sampling_percentage=0.1,
                               proportion=0.01, threshold=None, num_age_group=17, age_group_width=5,
                               data_store_path='./input/'):
        '''
        This is a method to generate the sampled result and plot a figure when considering non-responders
        This method should not be used directly, it is integrated within the __callable__ method

        '''
        if sampling_method == 'AgeRegion':  # Both age and region stratification

            # Only Random sample strategies
            infected_rate = []
            for i in range(len(time_sample)):  # Sample again at each time point
                if i == 0:
                    sampler_class = SamplerAgeRegion(data=self.demo_data, data_store_path=data_store_path,
                                                     num_age_group=num_age_group, age_group_width=age_group_width)
                else:
                    sampler_class = SamplerAgeRegion(data_store_path=data_store_path, pre_process=False,
                                                     num_age_group=num_age_group, age_group_width=age_group_width)
                try:
                    people = sampler_class.sample(sample_size=sample_size, additional_sample=additional_sample)
                except NameError:
                    people = sampler_class.sample(sample_size=sample_size)

                # Get the results of people sampled
                X = SamplingMaker(non_resp_rate=non_resp_rate, keeptrack=True, TheData=self.time_data,
                                  false_positive=0, false_negative=0, threshold=None)
                ite = X([time_sample[i]], people)

                # After each sample, now deal with the additional samples
                try:
                    # For the first time sampled, there is no variable additional sample defined,
                    # therefore, NameError would be raised
                    # Then it will go to the 'except NameError' line
                    additional_sample = np.array(additional_sample)
                    if additional_sample.sum() == 0:
                        # When there is no additional samples needed after one sample,
                        # we will also skip the following bit directly output the infection rate
                        # by raising the NameError
                        raise NameError
                    else:
                        # This means there are additional samples needed

                        # Record the position of groups that need additional samples
                        indices = np.nonzero(additional_sample)
                        add_pos = []
                        for k in range(len(indices[0])):
                            add_pos.append((indices[0][k], indices[1][k]))

                        # Now, we want to calculate the true infection rate
                        # But since we additionally sampled some people in some groups
                        # This will not be age-region stratification
                        # So we need to generate a robust infection rate according to
                        # The age-regional distribution
                        count_total = 0  # This is the total number of people from groups with addtional samples
                        count_posi = 0  # total number of positive from groups with additional samples
                        other_posi = 0  # total number of positive from groups without additional samples
                        count_nonResp = 0  # total number of non-responders from groups with additional samples
                        other_nonResp = 0  # total number of non-responders from groups without additional samples
                        # Now, we want to get the infection rate for each age-region group
                        # We need to firstly put each people sampled back to their original
                        # age-region group, and then identify whether they are in the group
                        # that with additional samples or not
                        for id in people:
                            region_pos = int(id.split('.')[0])
                            age_value = self.demo_data[self.demo_data['ID'] == id]['age'].values[0]
                            age_pos = min(num_age_group - 1, math.floor(age_value / age_group_width))
                            indexer = (region_pos, age_pos)
                            if indexer in add_pos:
                                # If this person is from a group with additional samples
                                count_total += 1
                                col_index = ite.columns.get_loc(id)
                                if ite.iloc[0, col_index] == 'Positive':
                                    count_posi += 1
                                if ite.iloc[0, col_index] == 'NonResponder':
                                    count_nonResp += 1
                            else:
                                # If this person is not from a group with additional samples
                                col_index = ite.columns.get_loc(id)
                                if ite.iloc[0, col_index] == 'Positive':
                                    other_posi += 1
                                if ite.iloc[0, col_index] == 'NonResponder':
                                    other_nonResp += 1

                        # The following is total number of people responded from groups with additional samples
                        effective_total = count_total - count_nonResp

                        if effective_total > 0:  # If there is one person respond
                            # We can then calculate the infected rate of groups with additional samples

                            # But do not forget that there are additional samples
                            # To maintain age-region stratification, we need to rescale to
                            # get age-region rescaled number of positive cases
                            spaces = sample_size - (len(people) - count_total)
                            spaces_posi = round(spaces * count_posi / effective_total)

                            # Then add positive cases from other groups(without additional samples)
                            # Then we can calculate this robust infected rate
                            infected_rate.append((spaces_posi + other_posi)
                                                 / (spaces + len(people) - count_total - other_nonResp))
                        else:  # If there is no one in these groups responded
                            try:
                                # Then try to use other groups' data to be the infected rate
                                infected_rate.append(other_posi / (len(people) - count_total - other_nonResp))
                            except ZeroDivisionError:
                                infected_rate.append(np.nan)
                except NameError:
                    # If additional_sample are not defined or sum = 0, i.e. No additional samples needed
                    # Then directly calculate the infected rate as the output
                    try:
                        infected_rate_ite = (ite.iloc[0].value_counts().get('Positive', 0)
                                             / (ite.iloc[0].value_counts().get('Positive', 0)
                                                + ite.iloc[0].value_counts().get('Negative', 0)))
                    except ZeroDivisionError:
                        # There is the possibility that all people do not respond,
                        # so just output nan
                        infected_rate_ite = np.nan
                    infected_rate.append(infected_rate_ite)

                # After each sample, we need to generate the additional samples for sampling next time
                # based on the non-responders' IDs of this time's sample
                non_resp_id = []
                for j in range(len(ite.columns)):
                    if ite.iloc[0, j] == 'NonResponder':
                        non_resp_id.append(ite.columns[j])
                additional_sample = sampler_class.additional_nonresponder(non_resp_id=non_resp_id,
                                                                          sampling_percentage=sampling_percentage,
                                                                          proportion=proportion, threshold=threshold)
        elif sampling_method == 'Age':  # Age stratification does not support non-responders
            raise ValueError('Age stratification method does not support non-responders, please disable non-responders')
        elif sampling_method == 'Region':  # Only with region stratification
            # Almost the same code as above, see comments above
            infected_rate = []
            for i in range(len(time_sample)):
                if i == 0:
                    sampler_class = SamplerRegion(data=self.demo_data, data_store_path=data_store_path)
                else:
                    sampler_class = SamplerRegion(data_store_path=data_store_path, pre_process=False)
                try:
                    people = sampler_class.sample(sample_size=sample_size, additional_sample=additional_sample)
                except NameError:
                    people = sampler_class.sample(sample_size=sample_size)
                X = SamplingMaker(non_resp_rate=non_resp_rate, keeptrack=True, TheData=self.time_data,
                                  false_positive=0, false_negative=0, threshold=None)
                ite = X([time_sample[i]], people)
                try:
                    additional_sample = np.array(additional_sample)
                    if additional_sample.sum() == 0:
                        raise NameError
                    else:
                        add_pos = np.nonzero(additional_sample)[0]
                        count_total = 0
                        count_posi = 0
                        other_posi = 0
                        count_nonResp = 0
                        other_nonResp = 0
                        for id in people:
                            indexer = int(id.split('.')[0])
                            if indexer in add_pos:
                                count_total += 1
                                col_index = ite.columns.get_loc(id)
                                if ite.iloc[0, col_index] == 'Positive':
                                    count_posi += 1
                                if ite.iloc[0, col_index] == 'NonResponder':
                                    count_nonResp += 1
                            else:
                                col_index = ite.columns.get_loc(id)
                                if ite.iloc[0, col_index] == 'Positive':
                                    other_posi += 1
                                if ite.iloc[0, col_index] == 'NonResponder':
                                    other_nonResp += 1
                        effective_total = count_total - count_nonResp
                        if effective_total > 0:
                            spaces = sample_size - (len(people) - count_total)
                            spaces_posi = round(spaces * count_posi / effective_total)
                            infected_rate.append((spaces_posi + other_posi)
                                                 / (spaces + len(people) - count_total - other_nonResp))
                        else:
                            try:
                                infected_rate.append(other_posi / (len(people) - count_total - other_nonResp))
                            except ZeroDivisionError:
                                infected_rate.append(np.nan)
                except NameError:
                    try:
                        infected_rate_ite = (ite.iloc[0].value_counts().get('Positive', 0)
                                             / (ite.iloc[0].value_counts().get('Positive', 0)
                                                + ite.iloc[0].value_counts().get('Negative', 0)))
                    except ZeroDivisionError:
                        infected_rate_ite = np.nan
                    infected_rate.append(infected_rate_ite)

                non_resp_id = []
                for j in range(len(ite.columns)):
                    if ite.iloc[0, j] == 'NonResponder':
                        non_resp_id.append(ite.columns[j])
                additional_sample = sampler_class.additional_nonresponder(non_resp_id=non_resp_id,
                                                                          sampling_percentage=sampling_percentage,
                                                                          proportion=proportion, threshold=threshold)
        elif sampling_method == 'Base':  # Completely random sampling does not support non-responders
            raise ValueError('Random sampling does not support non-responders, please disable non-responders')
        else:
            raise ValueError('You must input a valid sampling method')

        # This is to generate the plot
        if gen_plot:
            plt.plot(time_sample, infected_rate)
            plt.xlabel('Time')
            plt.ylabel('Proportion of population')
            plt.xlim(0, max(time_sample))
            plt.ylim(0, 1)
            plt.title('Proportion of infection in the sample (consider non-responders)')
            if saving_path_sampling:
                plt.savefig(saving_path_sampling)
        res = []
        res.append(time_sample)
        res.append(infected_rate)
        # Store the result for comparison use
        self.result = infected_rate
        return res

    def _compare(self, time_sample, gen_plot=False, scale_method: str = 'proportional', saving_path_compare=None):
        '''
        Generate a graph comparing the difference between predicted and real infection level
        This method should not be used directly, it is integrated within the __callable__ method

        '''
        # Based on the input, use different scale method to estimate the true infection number
        if scale_method == 'proportional':
            result_scaled = np.array(self.result) * len(self.demo_data)

        # Get the true result from self.time_data
        true_result = []
        for t in time_sample:
            num = self.time_data.iloc[t, 1:].value_counts().get(3, 0)
            num += self.time_data.iloc[t, 1:].value_counts().get(4, 0)
            num += self.time_data.iloc[t, 1:].value_counts().get(5, 0)
            num += self.time_data.iloc[t, 1:].value_counts().get(6, 0)
            num += self.time_data.iloc[t, 1:].value_counts().get(7, 0)
            num += self.time_data.iloc[t, 1:].value_counts().get(8, 0)
            true_result.append(num)

        # Find the difference between estimated infection level and the real one
        diff = np.array(true_result) - result_scaled
        if gen_plot:
            plt.plot(time_sample, result_scaled, label='Predicted result', linestyle='--')
            plt.plot(time_sample, true_result, label='True result')
            plt.plot(time_sample, diff, label='Difference')
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('Population')
            plt.title('Number of infection in the population')
            if saving_path_compare:
                plt.savefig(saving_path_compare)
        return diff

    def _diff_processing(self, diff, metric):
        '''
        Function for transforming the diff into a value according to different metric

        Parameters:
        -----------

        diff : list
            The diff from _compare method
        metric : str
            A specific string specifying the method used to transform

        Output:
        -------

        A float number
        '''
        if metric == 'mean':
            return np.nanmean(np.abs(diff))
        elif metric == 'max':
            return max(np.abs(diff))

    def _iteration_once(
            self,
            sampling_interval,
            total_day_number,
            non_responder,
            hyperparameter_autotune,
            recognised_methods,
            sample_size,
            useful_inputs,
            metric,
            data_store_path=None,
            job_id=None,
            temp_folder_name=None,
            non_resp_rate=None,
            useful_inputs_nonrespRange=None
    ):
        '''
        The function to perform one iteration

        Parameters:
        -----------

        sampling_interval : int
            The number of days between two sample time points
        total_day_number : int
            The total number of days that simulated
        non_responder : bool
            Whether or not to consider non-responders
        hyperparameter_autotune : bool
            Whether or not to autotune the hyperparameters
        recognised_methods : list
            A list of sampling methods that is recognised by 'best_method' method
        sample_size : int
            The size of the sample
        useful_inputs : dict
            A dictionary including all parameters used for sampling
        metric : str
            A specific string indicating the metric used to transform diff to a single value
        job_id : int
            An ID of the current job when multiprocessing is on,
            when this value is None, it means the multiprocessing is off.
            When we turn on multiprocessing, a value will be passed to this parameter.

            Default = None
        temp_folder_name : str
            The name of the folder to store the files generated, it will be cleaned after.
            This is set to None by default, when we need multiprocessing, a value will be
            passed to this parameter

            Default = None
        non_resp_rate : float between 0 and 1
            The possibility of a person to be non-responders.
            When we consider non-responders, a value will be passed to this parameter.

            Default = None
        useful_inputs_nonrespRange : dict
            When hyperparameter tuning is on, and non-responder is on, the 'Region' method requires different input.
            This dictionary include these inputs. When we consider non-responders, a value will be passed
            to this parameter.

            Default = None

        Output:
        -------

        results : list of list
            A list of n lists, where n is the total number of recognised methods.
            Each list within 'results' contain the results of the same method under different sets of parameters.
            *The length of these lists are not the same since the number of combinations of parameters are different.*
        '''
        # Create a temperary folder to put temperary files under the path of __main__ files
        if job_id is not None:
            main_module_path = os.path.abspath(sys.modules['__main__'].__file__)
            dir_name = os.path.dirname(main_module_path)
            data_store_path = os.path.join(dir_name, temp_folder_name, 'job_id_' + str(job_id))
            os.mkdir(data_store_path)
        else:
            data_store_path = data_store_path

        # Firstly define the time points to sample based on sampling-interval
        time_sample = list(np.arange(math.floor(total_day_number / sampling_interval))
                           * sampling_interval)

        # Divide into different cases
        if non_responder is False:
            if hyperparameter_autotune is False:

                # This is the result to output in the end, performance of different methods
                res_across_methods = []
                for method in recognised_methods:

                    # Put the result of the same method into one list
                    result_within_method = []

                    # Split the method name and sample strategy
                    method_string = method.split('-')
                    if method_string[1] == 'Same':
                        input_kwargs = {
                            'sample_strategy': 'Same'
                        }

                        # Pour the inputs in useful_inputs into the dict to input
                        for input in useful_inputs:
                            input_kwargs[input] = useful_inputs[input]

                        # Perform the sampling by __call__ method above
                        _, diff = self(method_string[0], sample_size,
                                       time_sample, data_store_path=data_store_path,
                                       **input_kwargs)

                        # Process the diff according to the metric provided
                        result_within_method.append(self._diff_processing(diff, metric))
                    elif method_string[1] == 'Random':
                        # The following part is almost same as above
                        input_kwargs = {
                            'sample_strategy': 'Random'
                        }
                        for input in useful_inputs:
                            input_kwargs[input] = useful_inputs[input]
                        _, diff = self(method_string[0], sample_size,
                                       time_sample, data_store_path=data_store_path,
                                       **input_kwargs)
                        result_within_method.append(self._diff_processing(diff, metric))

                    # For different methods, we have a list to contain its result,
                    # I do this because there will be different parameter pairs to repeat
                    res_across_methods.append(result_within_method)

                # Output the final result and clean up
                if job_id is not None:
                    self._clean_up(temp_folder_name=temp_folder_name, data_store_path=data_store_path)
                return res_across_methods
            else:

                # This is the code when considering hyperparameter autotuning
                # Use the same structure as above
                # Comment when there is something is different
                res_across_methods = []
                for method in recognised_methods:
                    result_within_method = []
                    method_string = method.split('-')
                    if method_string[1] == 'Same':
                        input_kwargs = {
                            'sample_strategy': 'Same'
                        }

                        # Here, we need to distuiguish between Age-related and
                        # Age-unrelated, since the inputs are different
                        # For age-related, they need the num_age_group and
                        # age_group_width variables
                        if method_string[0] == 'Base' or method_string[0] == 'Region':

                            # Since there is no parameters to vary,
                            # So just like above, directly output
                            # the result
                            _, diff = self(method_string[0], sample_size,
                                           time_sample, data_store_path=data_store_path,
                                           **input_kwargs)
                            result_within_method.append(self._diff_processing(diff, metric))
                        elif method_string[0] == 'Age' or method_string[0] == 'AgeRegion':

                            # Now we have parameters to vary
                            # Firstly we should collect all parameters can vary
                            # And put their ranges into a list
                            all_ranges = []
                            for key in useful_inputs:
                                all_ranges.append(useful_inputs[key])

                            # Use this list to generate all possible combinations
                            # of different parameters
                            all_combinations = list(product(*all_ranges))

                            # For each combination, do a sampling and output result
                            for combination in all_combinations:
                                count = 0
                                for key in useful_inputs:
                                    input_kwargs[key[:-6]] = combination[count]
                                    count += 1
                                _, diff = self(method_string[0], sample_size,
                                               time_sample, data_store_path=data_store_path,
                                               **input_kwargs)
                                result_within_method.append(self._diff_processing(diff, metric))
                    elif method_string[1] == 'Random':
                        # Exactly the same logic as above
                        input_kwargs = {
                            'sample_strategy': 'Random'
                        }
                        if method_string[0] == 'Base' or method_string[0] == 'Region':
                            _, diff = self(method_string[0], sample_size,
                                           time_sample, data_store_path=data_store_path,
                                           **input_kwargs)
                            result_within_method.append(self._diff_processing(diff, metric))
                        elif method_string[0] == 'Age' or method_string[0] == 'AgeRegion':
                            all_ranges = []
                            for key in useful_inputs:
                                all_ranges.append(useful_inputs[key])
                            all_combinations = list(product(*all_ranges))
                            for combination in all_combinations:
                                count = 0
                                for key in useful_inputs:
                                    input_kwargs[key[:-6]] = combination[count]
                                    count += 1
                                _, diff = self(method_string[0], sample_size,
                                               time_sample, data_store_path=data_store_path,
                                               **input_kwargs)
                                result_within_method.append(self._diff_processing(diff, metric))
                    res_across_methods.append(result_within_method)
                if job_id is not None:
                    self._clean_up(temp_folder_name=temp_folder_name, data_store_path=data_store_path)
                return res_across_methods
        else:

            # The following part is for the case considering non-responders
            # Use the same logic as above
            # Comments when there is something different below
            if hyperparameter_autotune is False:
                res_across_methods = []
                for method in recognised_methods:
                    result_within_method = []
                    method_string = method.split('-')
                    input_kwargs = {}
                    for input in useful_inputs:
                        input_kwargs[input] = useful_inputs[input]
                    _, diff = self(method_string[0], sample_size,
                                   time_sample, non_responder=True,
                                   non_resp_rate=non_resp_rate, data_store_path=data_store_path,
                                   **input_kwargs)
                    result_within_method.append(self._diff_processing(diff, metric))
                    res_across_methods.append(result_within_method)
                if job_id is not None:
                    self._clean_up(temp_folder_name=temp_folder_name, data_store_path=data_store_path)
                return res_across_methods
            else:
                res_across_methods = []
                for method in recognised_methods:
                    result_within_method = []
                    method_string = method.split('-')
                    input_kwargs = {}
                    if method_string[0] == 'Region':

                        # Here has something different
                        # Since Region method does not have the num_age_group
                        # and age_group_width variable to vary,
                        # So we need to use a different useful_input dict
                        # to setup the ranges to generate the combinations
                        all_ranges = []
                        for key in useful_inputs_nonrespRange:
                            all_ranges.append(useful_inputs_nonrespRange[key])
                        all_combinations = list(product(*all_ranges))
                        for combination in all_combinations:
                            count = 0
                            for key in useful_inputs_nonrespRange:
                                input_kwargs[key[:-6]] = combination[count]
                                count += 1
                            _, diff = self(method_string[0], sample_size,
                                           time_sample, non_responder=True,
                                           non_resp_rate=non_resp_rate, data_store_path=data_store_path,
                                           **input_kwargs)
                            result_within_method.append(self._diff_processing(diff, metric))
                    elif method_string[0] == 'AgeRegion':

                        # Here is just the normal case
                        all_ranges = []
                        for key in useful_inputs:
                            all_ranges.append(useful_inputs[key])
                        all_combinations = list(product(*all_ranges))
                        for combination in all_combinations:
                            count = 0
                            for key in useful_inputs:
                                input_kwargs[key[:-6]] = combination[count]
                                count += 1
                            _, diff = self(method_string[0], sample_size,
                                           time_sample, non_responder=True,
                                           non_resp_rate=non_resp_rate, data_store_path=data_store_path,
                                           **input_kwargs)
                            result_within_method.append(self._diff_processing(diff, metric))
                    res_across_methods.append(result_within_method)
                if job_id is not None:
                    self._clean_up(temp_folder_name=temp_folder_name, data_store_path=data_store_path)
                return res_across_methods

    def _clean_up(self, temp_folder_name, data_store_path):
        '''
        This method is to clean up the temporary files generated during multiprocessing.
        This is called within the method '_iteration_once' when multiprocessing is on.

        Parameters:
        -----------

        temp_folder_name : str
            The name of the file to store the files
        data_store_path : str
            The exact path to store the data
        '''
        # This is to locate the path from __main__
        # This part can be unnecessary, but find the path from absolute path
        # is to avoid weird names in some path names
        main_module_path = os.path.abspath(sys.modules['__main__'].__file__)
        dir_name = os.path.dirname(main_module_path)

        # From here, removing the job_id_i folder and its content
        if os.path.exists(os.path.join(dir_name, temp_folder_name)):
            if os.path.exists(data_store_path):
                for file in ['pop_dist.json', 'microcells.csv', 'data.csv']:
                    if os.path.exists(data_store_path + file):
                        os.remove(data_store_path + file)
                os.rmdir(data_store_path)

    def _wrapper_iteration_once(self, kwargs_dict):
        '''
        Since the input variables of _iteration_once may be different, need this function to wrap up these inputs

        Parameters:
        -----------

        kwargs_dict : dict
            A dictionary of inputs of '_iteration_once'
        '''
        return self._iteration_once(**kwargs_dict)

    def best_method(self, methods, sample_size, hyperparameter_autotune=False,
                    non_responder=False, non_resp_rate=None, sampling_interval=7,
                    parallel_computation=True, metric='mean', iteration=100,
                    data_store_path='./input/', **kwargs):
        '''
        Print the best method among different methods provided.

        When hyper-parameter autotune is on, will firstly print the best parameter
        combination and its performance of each method, then print the best method
        across all methods.

        The order of best parameter set printed follows the following ordering:
            (
            'num_age_group',
            'age_group_width',
            'sampling_percentage',
            'proportion',
            'threshold'
            )

            Parameter will be omitted if that parameter is not applicable for the
            method.

        Features:
        ---------

        When a range of parameters provided, can automatically tune the hyperparameters

        Can set to consider non-ressponders

        Will print any unrecognised inputs or methods

        Parameters:
        -----------

        methods : list
            A list of strings indicating the methods to compare with each other
            Acceptible methods:
                Use 'Same' strategy:
                    'Age-Same'
                    'Region-Same'
                    'AgeRegion-Same'
                    'Base-Same'
                Use 'Random' strategy:
                    'Age-Random'
                    'Region-Random'
                    'AgeRegion-Random'
                    'Base-Random'
                *Note: When you input the method names without sample strategy, 'Random' will be the default*
        sample_size : int
            The size of sample
        hyperparameter_autotune : bool
            Whether or not to turn on the hyperparameter automatic tuning

            *For extra input, see documentation for parameter 'kwargs' below*
        non_responder : bool
            Whether or not to consider non-responders
        sampling_interval : int
            The number of days between each sampling time points
        metric : str
            The metric used to transform difference between the sampled result and true infection into
            a float to measure the performance.
            Acceptible metric:
                'mean':
                    Use the mean of absolute difference between true and predicted infection.
                    We ignore all nan values
                'max':
                    Use the max of absolute difference between true and predicted infection.
        iteration : int
            The number of iterations to run and average the value of prediction to get
            a robust result
        parallel_computation : bool
            Whether or not to use multiprocessing to speed up this repeated process

            Default = True

            *Note: You cannot directly call this method when this is turned on, see example in documentation*
        data_store_path : str
            The path to store files generated during sampling when parallel computation is disabled

            Default = './input/'

            *This is used only when parallel computation is disabled*
        kwargs : dict
            A dictionary of parameters passed to process part
            The following parameters can be passed:
                num_age_group : int
                    Indicating how many age groups are there.

                    Default = 17

                    *The last group includes age >= some threshold*

                    *This is used when autotuning is turned off*
                age_group_width : int
                    Indicating the width of each age group(except for the last group)

                    Default = 5

                    *This is used when autotuning is turned off*
                sampling_percentage : float, between 0 and 1
                    The proportion of additional samples taken from a specific (age-)regional group

                    Default = 0.1 (Only for non-responders)

                    *This is used when autotuning is turned off*
                proportion : float, between 0 and 1
                    The proportion of total groups to be sampled additionally

                    Default = 0.01 (Only for non-responders)

                    *This is used when autotuning is turned off*
                threshold : NoneType or Int
                    The lowest number of groups to be sampled additionally

                    Default = None (Only for non-responders)

                    *This is used when autotuning is turned off*
                num_age_group_range : list
                    All possible number of age groups that you want to try/iterate over

                    Default = [10, 13, 15, 17, 20]

                    *The last group includes age >= some threshold*

                    *This is used when autotuning is turned on*
                age_group_width_range : list
                    All possible age group width (except for the last group)
                    that you want to try/iterate over

                    Default = [5, 10]

                    *This is used when autotuning is turned on*
                sampling_percentage_range : list
                    All possible proportion of additional samples taken
                    from a specific (age-)regional group that you want to
                    try/iterate over

                    Default = [0.1, 0.2, 0.3] (Only for non-responders)

                    *This is used when autotuning is turned on*
                proportion_range : list
                    All possible proportion of total groups to be sampled additionally
                    that you want to try/iterate over

                    Default = [0.01, 0.05, 0.1] (Only for non-responders)

                    *This is used when autotuning is turned on*
                threshold_range : list
                    All possible lowest number of groups to be sampled additionally
                    that you want to try/iterate over

                    Default = [10, 20, 30] (Only for non-responders)

                    *This is used when autotuning is turned on*
        '''
        # Check whether metric is recognisable
        recognisable_metric = [
            'mean',
            'max'
        ]
        if metric not in recognisable_metric:
            raise ValueError('Metric not recognisable')

        # All recognisable inputs
        recognisable_inputs = [
            'num_age_group',
            'age_group_width',
            'sampling_percentage',
            'proportion',
            'threshold',
            'num_age_group_range',
            'age_group_width_range',
            'sampling_percentage_range',
            'proportion_range',
            'threshold_range'
        ]

        # All parameters used when the autotune function enabled
        inputs_for_autotune_normal = {
            'num_age_group_range': [10, 13, 15, 17, 20],
            'age_group_width_range': [5, 10]
        }
        inputs_for_autotune_nonresp = {
            'num_age_group_range': [10, 13, 15, 17, 20],
            'age_group_width_range': [5, 10],
            'sampling_percentage_range': [0.1, 0.2, 0.3],
            'proportion_range': [0.01, 0.05, 0.1],
            'threshold_range': [10, 20, 30]
        }
        inputs_for_autotune_nonresp_Region = {
            'sampling_percentage_range': [0.1, 0.2, 0.3],
            'proportion_range': [0.01, 0.05, 0.1],
            'threshold_range': [10, 20, 30]
        }

        # All parameters used when the autotune function disabled
        inputs_for_disabled_autotune_normal = [
            'num_age_group',
            'age_group_width'
        ]
        inputs_for_disabled_autotune_nonresp = [
            'num_age_group',
            'age_group_width',
            'sampling_percentage',
            'proportion',
            'threshold'
        ]

        # Parameter Ordering when hyperparameter autotune is on
        inputs_ordering_normal = [
            'num_age_group_range',
            'age_group_width_range'
        ]
        inputs_ordering_nonrespRegion = [
            'sampling_percentage_range',
            'proportion_range',
            'threshold_range'
        ]
        inputs_ordering_nonrespAgeRegion = [
            'num_age_group_range',
            'age_group_width_range',
            'sampling_percentage_range',
            'proportion_range',
            'threshold_range'
        ]

        # Firstly remove all irrecognisable inputs
        recognised_inputs = {}
        irrecognisable_input = []
        for input in kwargs:
            if input in recognisable_inputs:
                recognised_inputs[input] = kwargs[input]
            else:
                irrecognisable_input.append(input)

        # Then selected the useful inputs for the function enabled
        useful_inputs = {}
        if hyperparameter_autotune:
            if non_responder is False:

                # Put all specified recognised inputs into useful inputs
                # Discard the others
                for input in recognised_inputs:
                    if input in inputs_for_autotune_normal:
                        useful_inputs[input] = recognised_inputs[input]
                    else:
                        irrecognisable_input.append(input)

                # When some of the inputs are not specified, use default values stored
                for input in inputs_for_autotune_normal:
                    if input in useful_inputs:
                        pass
                    else:
                        useful_inputs[input] = inputs_for_autotune_normal[input]
            else:

                # Since 'Region' do not need to iterate over age stuffs, define a new
                # dict to contain these variables
                useful_inputs_nonrespRange = {}

                # Put all specified recognised inputs into useful inputs
                # Discard the others
                for input in recognised_inputs:
                    if input in inputs_for_autotune_nonresp:
                        useful_inputs[input] = recognised_inputs[input]
                    else:
                        irrecognisable_input.append(input)

                    # Do the same for Region
                    if input in inputs_for_autotune_nonresp_Region:
                        useful_inputs_nonrespRange[input] = recognised_inputs[input]

                # When some of the inputs are not specified, use default values stored
                for input in inputs_for_autotune_nonresp:
                    if input in useful_inputs:
                        pass
                    else:
                        useful_inputs[input] = inputs_for_autotune_nonresp[input]

                # Do the same for Region
                for input in inputs_for_autotune_nonresp_Region:
                    if input in useful_inputs_nonrespRange:
                        pass
                    else:
                        useful_inputs_nonrespRange[input] = inputs_for_autotune_nonresp_Region[input]
        else:
            if non_responder:  # Since non-responders will need more inputs, divide into two situations
                for input in recognised_inputs:
                    if input in inputs_for_disabled_autotune_nonresp:
                        useful_inputs[input] = recognised_inputs[input]
                    else:
                        irrecognisable_input.append(input)
            else:
                for input in recognised_inputs:
                    if input in inputs_for_disabled_autotune_normal:
                        useful_inputs[input] = recognised_inputs[input]
                    else:
                        irrecognisable_input.append(input)

        # Print the unused inputs
        if irrecognisable_input:
            print_str = 'The following inputs provided are not used: '
            print_str += ', '.join(irrecognisable_input)
            print(print_str)

        total_day_number = len(self.time_data)

        # For methods, need to distinguish recognised ones
        recognisable_methods = [
            'AgeRegion',
            'Age',
            'Region',
            'Base',
            'AgeRegion-Same',
            'Age-Same',
            'Region-Same',
            'Base-Same',
            'AgeRegion-Random',
            'Age-Random',
            'Region-Random',
            'Base-Random'
        ]
        recognised_methods = set()
        irrecognisable = []
        for method in methods:
            if method in recognisable_methods:
                if len(method.split('-')) == 1:
                    recognised_methods.add(method + '-Random')
                else:
                    recognised_methods.add(method)
            else:
                irrecognisable.append(method)

        # If non-responder function enabled, then all 'Same' method will be ignored
        if non_responder:
            for method in recognised_methods.copy():
                if method[-4:] == 'Same':
                    recognised_methods.remove(method)
                    irrecognisable.append(method)
                elif method[:-7] == 'Age' or method[:-7] == 'Base':
                    recognised_methods.remove(method)
                    irrecognisable.append(method)

        # Print all irrecognised methods
        if irrecognisable:
            print_str = 'The following methods provided are not used: '
            for i in irrecognisable:
                print_str += i
                print_str += ', '
            print_str = print_str[:-2]
            print(print_str)

        recognised_methods = list(recognised_methods)
        if len(recognised_methods) == 0:
            raise ValueError('No valid method detected')

        # Prepare the inputs for each iteration
        iteration_inputs = {
            'sampling_interval': sampling_interval,
            'total_day_number': total_day_number,
            'non_responder': non_responder,
            'hyperparameter_autotune': hyperparameter_autotune,
            'recognised_methods': recognised_methods,
            'sample_size': sample_size,
            'useful_inputs': useful_inputs,
            'metric': metric
        }
        if non_responder:
            iteration_inputs['non_resp_rate'] = non_resp_rate
            if hyperparameter_autotune:
                if 'Region-Random' in recognised_methods:
                    iteration_inputs['useful_inputs_nonrespRange'] = useful_inputs_nonrespRange

        # Prepare the folder name to store the temperary data
        if parallel_computation:
            temp_folder_name = 'temp_'
            main_module_path = os.path.abspath(sys.modules['__main__'].__file__)
            dir_name = os.path.dirname(main_module_path)
            while os.path.exists(os.path.join(dir_name, temp_folder_name)):
                temp_folder_name += 'a'
            iteration_inputs['temp_folder_name'] = temp_folder_name
            os.mkdir(os.path.join(dir_name, temp_folder_name))

            # From here, enable multiprocessing
            # Firstly, prepare the input
            multiprocessing_inputs = []
            for i in range(iteration):
                multiprocessing_input = iteration_inputs.copy()
                multiprocessing_input['job_id'] = i
                multiprocessing_inputs.append(multiprocessing_input)
            num_processes = multiprocessing.cpu_count()
            with multiprocessing.Pool(processes=num_processes) as pool:
                # Map the process_item function to the items
                results = pool.map(self._wrapper_iteration_once, multiprocessing_inputs)

            if os.path.exists(os.path.join(dir_name, temp_folder_name)):
                os.rmdir(os.path.join(dir_name, temp_folder_name))
        else:
            # Here is when the multiprocessing is disabled
            iteration_inputs['data_store_path'] = data_store_path
            results = []
            for i in range(iteration):
                results.append(self._iteration_once(**iteration_inputs))

        # Average the result over all iterations
        res = []
        for j in range(len(results[0])):
            res.append([])
            for i in range(len(results[0][j])):
                list_ite = []
                for k in range(iteration):
                    list_ite.append(results[k][j][i])
                res[j].append(np.nanmean(list_ite))

        # This last block is to print out the result
        if hyperparameter_autotune is False:
            # When autotune is off, each row only have one element
            # So we can directly find min and print
            output = []
            for i in res:
                output += i
            min_index = output.index(min(output))
            print('The best method is %s, with %s difference %s' % (recognised_methods[min_index],
                                                                    metric, output[min_index]))
        else:
            # When autotune is on, each row have many elements
            # Need to firstly find the min in each row(each method)
            output = {}
            for i in range(len(res)):
                min_index = res[i].index(min(res[i]))
                output[i] = res[i][min_index]
                if non_responder is False:
                    method = recognised_methods[i]
                    method_string = method.split('-')

                    # Print out the best combination parameters for different methods
                    if method_string[0] == 'Base' or method_string[0] == 'Region':
                        print('%s method has %s difference %s' % (method, metric, res[i][min_index]))
                    elif method_string[0] == 'Age' or method_string[0] == 'AgeRegion':
                        all_ranges = []
                        ordering = []
                        for key in useful_inputs:
                            all_ranges.append(useful_inputs[key])
                            ordering.append(inputs_ordering_normal.index(key))
                        all_combinations = list(product(*all_ranges))
                        best_parameter_value = all_combinations[min_index]
                        ordered_best_parameter_value = []
                        for position in range(len(best_parameter_value)):
                            reorder_position = ordering.index(position)
                            ordered_best_parameter_value.append(best_parameter_value[reorder_position])
                        print('The best %s method achieved when parameter is %s, with %s difference %s'
                              % (method, tuple(ordered_best_parameter_value), metric, res[i][min_index]))

                else:
                    method = recognised_methods[i].split('-')[0]

                    # Print out the best combination parameters for different methods
                    if method == 'Region':
                        all_ranges = []
                        ordering = []
                        for key in useful_inputs_nonrespRange:
                            all_ranges.append(useful_inputs_nonrespRange[key])
                            ordering.append(inputs_ordering_nonrespRegion.index(key))
                        all_combinations = list(product(*all_ranges))
                        best_parameter_value = all_combinations[min_index]
                        ordered_best_parameter_value = []
                        for position in range(len(best_parameter_value)):
                            reorder_position = ordering.index(position)
                            ordered_best_parameter_value.append(best_parameter_value[reorder_position])
                        print('The best %s method achieved when parameter is %s, with %s difference %s'
                              % (method, tuple(ordered_best_parameter_value), metric, res[i][min_index]))
                    elif method == 'AgeRegion':
                        all_ranges = []
                        ordering = []
                        for key in useful_inputs:
                            all_ranges.append(useful_inputs[key])
                            ordering.append(inputs_ordering_nonrespAgeRegion.index(key))
                        all_combinations = list(product(*all_ranges))
                        best_parameter_value = all_combinations[min_index]
                        ordered_best_parameter_value = []
                        for position in range(len(best_parameter_value)):
                            reorder_position = ordering.index(position)
                            ordered_best_parameter_value.append(best_parameter_value[reorder_position])
                        print('The best %s method achieved when parameter is %s, with %s difference %s'
                              % (method, tuple(ordered_best_parameter_value), metric, res[i][min_index]))

            # Find the best method among all methods, and print it out
            min_index = min(output)
            min_value = min(output.values())
            print('The best method is %s, with %s difference %s' % (recognised_methods[min_index], metric, min_value))
