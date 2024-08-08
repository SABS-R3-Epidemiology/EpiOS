import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from epios import SamplingMaker

class BaseFastSampler():

    def __init__(self, data: pd.DataFrame) -> None:

        self.data = data
    
    def sample(self, sample_size: int, sampling_seed: int =None) -> list:

        if sample_size > len(self.data):
            raise ValueError('Sample size is larger than the input data size.')
        if sampling_seed is not None:
            np.random.seed(sampling_seed)
        return self.data['id'].sample(n=sample_size, replace=False).tolist()


class AgeRegionFastSampler(BaseFastSampler):

    def __init__(self, data: pd.DataFrame, num_age_group: int = 17, age_group_width: int = 5) -> None:

        super().__init__(data)
        self.num_age_group = num_age_group
        self.age_group_width = age_group_width
    
    def sample(self, sample_size: int, sampling_seed: int = None) -> list:

        n = len(self.data)
        if sample_size > n:
            raise ValueError('Sample size is larger than the input data size.')
        if sampling_seed is not None:
            np.random.seed(sampling_seed)
        # age_dist = self.data['age'].value_counts(normalize=True)
        region_holder = self.data['id'].tolist()
        region_holder = [int(x.split('.')[0]) for x in region_holder]
        num_region = max(region_holder) + 1
        ar_cap = np.zeros((num_region, self.num_age_group))
        id_list = {}
        for _, row in self.data.iterrows():
            region = int(row['id'].split('.')[0])
            age = row['age']
            age_group = int(age // self.age_group_width)
            if age_group >= self.num_age_group:
                age_group = self.num_age_group - 1
            ar_cap[region, age_group] += 1
            id_list[(region, age_group)] = id_list.get((region, age_group), []) + [row['id']]
        ar_cap = ar_cap.flatten()
        ar_dist = ar_cap / n
        num_samples = np.floor(sample_size * ar_dist)
        num_samples = num_samples.astype(int)

        remaining_samples = sample_size - num_samples.sum()
        # remaining_dist = {}
        while remaining_samples > 0:
            remaining_indices = np.where(ar_cap > num_samples)[0]
            num_remaining_cat = len(remaining_indices)
            if num_remaining_cat >= remaining_samples:
                for ind in np.random.choice(remaining_indices, remaining_samples, replace=False):
                    # remaining_dist[ind] = remaining_dist.get(ind, 0) + 1
                    num_samples[ind] += 1
                remaining_samples = 0
            else:
                for ind in remaining_indices:
                    # remaining_dist[ind] = remaining_dist.get(ind, 0) + 1
                    num_samples[ind] += 1
                remaining_samples -= num_remaining_cat
        
        num_samples = num_samples.reshape((num_region, self.num_age_group))
        sample_list = []
        for i in range(num_region):
            for j in range(self.num_age_group):
                if num_samples[i, j] > 0:
                    sample_list += np.random.choice(id_list[(i, j)], num_samples[i, j], replace=False).tolist()
        return sample_list


class AgeFastSampler(BaseFastSampler):

    def __init__(self, data: pd.DataFrame, num_age_group: int = 17, age_group_width: int = 5) -> None:

        super().__init__(data)
        self.num_age_group = num_age_group
        self.age_group_width = age_group_width
    
    def sample(self, sample_size: int, sampling_seed: int = None) -> list:

        n = len(self.data)
        if sample_size > n:
            raise ValueError('Sample size is larger than the input data size.')
        if sampling_seed is not None:
            np.random.seed(sampling_seed)
        # age_dist = self.data['age'].value_counts(normalize=True)
        ar_cap = np.zeros(self.num_age_group)
        id_list = {}
        for row in self.data.itertuples():
            age = row.age
            age_group = int(age // self.age_group_width)
            if age_group >= self.num_age_group:
                age_group = self.num_age_group - 1
            ar_cap[age_group] += 1
            id_list[age_group] = id_list.get(age_group, []) + [row.id]
        ar_dist = ar_cap / n
        num_samples = np.floor(sample_size * ar_dist)
        num_samples = num_samples.astype(int)

        remaining_samples = sample_size - num_samples.sum()
        # remaining_dist = {}
        while remaining_samples > 0:
            remaining_indices = np.where(ar_cap > num_samples)[0]
            num_remaining_cat = len(remaining_indices)
            if num_remaining_cat >= remaining_samples:
                for ind in np.random.choice(remaining_indices, remaining_samples, replace=False):
                    # remaining_dist[ind] = remaining_dist.get(ind, 0) + 1
                    num_samples[ind] += 1
                remaining_samples = 0
            else:
                for ind in remaining_indices:
                    # remaining_dist[ind] = remaining_dist.get(ind, 0) + 1
                    num_samples[ind] += 1
                remaining_samples -= num_remaining_cat

        sample_list = []
        for j in range(self.num_age_group):
            if num_samples[j] > 0:
                sample_list += np.random.choice(id_list[j], num_samples[j], replace=False).tolist()
        return sample_list


class RegionFastSampler(BaseFastSampler):

    def __init__(self, data: pd.DataFrame) -> None:

        super().__init__(data)

    def sample_old(self, sample_size: int, sampling_seed: int = None) -> list:

        n = len(self.data)
        if sample_size > n:
            raise ValueError('Sample size is larger than the input data size.')
        if sampling_seed is not None:
            np.random.seed(sampling_seed)
        # age_dist = self.data['age'].value_counts(normalize=True)
        region_holder = self.data['id'].tolist()
        region_holder = [int(x.split('.')[0]) for x in region_holder]
        num_region = max(region_holder) + 1
        ar_cap = np.zeros(num_region)
        id_list = {}
        for row in self.data.itertuples():
            region = int(row.id.split('.')[0])
            ar_cap[region] += 1
            id_list[region] = id_list.get(region, []) + [row.id]
        ar_dist = ar_cap / n
        num_samples = np.floor(sample_size * ar_dist)
        num_samples = num_samples.astype(int)

        remaining_samples = sample_size - num_samples.sum()
        # remaining_dist = {}
        while remaining_samples > 0:
            remaining_indices = np.where(ar_cap > num_samples)[0]
            num_remaining_cat = len(remaining_indices)
            if num_remaining_cat >= remaining_samples:
                for ind in np.random.choice(remaining_indices, remaining_samples, replace=False):
                    # remaining_dist[ind] = remaining_dist.get(ind, 0) + 1
                    num_samples[ind] += 1
                remaining_samples = 0
            else:
                for ind in remaining_indices:
                    # remaining_dist[ind] = remaining_dist.get(ind, 0) + 1
                    num_samples[ind] += 1
                remaining_samples -= num_remaining_cat
        
        num_samples = num_samples.reshape(num_region)
        sample_list = []
        for i in range(num_region):
            if num_samples[i] > 0:
                sample_list += np.random.choice(id_list[i], num_samples[i], replace=False).tolist()
        return sample_list

    def sample(self, sample_size: int, sampling_seed: int = None) -> list:
        # KG - Uses builtin pandas methods to hopefully speed up the previous version

        if sampling_seed is not None:
            np.random.seed(sampling_seed)

        # Calculate the count of each region
        region_counts = self.data['region'].value_counts().sort_index()
        total_count = len(self.data)

        # Determine the number of samples per region proportionally
        region_proportions = region_counts / total_count
        num_samples = (region_proportions * sample_size).astype(int)

        # Handle remaining samples due to flooring
        remaining_samples = sample_size - num_samples.sum()
        if remaining_samples > 0:
            extra_samples_regions = region_counts[region_counts > num_samples].index
            extra_samples_allocation = np.random.choice(extra_samples_regions, remaining_samples, replace=False)
            num_samples.loc[extra_samples_allocation] += 1

        # Sample the required number of IDs from each region
        sampled_ids = []
        for region, count in num_samples.items():
            if count > 0:
                region_ids = self.data[self.data['region'] == region]['id']
                sampled_ids.extend(region_ids.sample(n=count, replace=False).tolist())

        return sampled_ids
    

class FastPostProcess():
    '''
    This class is to automatically sample the population at several given time points.

    And generate plots and comparison with the true infection level within the population.

    How to use:
    -----------

    Define an instance and input the demographical and time data of the population
    Then use self.predict to generate plots and comparison

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
        self.predict = self.Prediction(demo_data=demo_data, time_data=time_data)

    class Prediction():
        '''
        This sub-class is to automatically sample the population at several given time points.

        This sub-class is automatically defined when an instance of PostProcess is defined.

        To use this class, call methods defined under this class to sample and generate plots.

        Parameters:
        -----------

        demo_data : pandas.DataFrame
            The geographical data of the population
        time_data : pandas.DataFrame
            The infection data of the population at different time points

        '''

        def __init__(self, demo_data: pd.DataFrame, time_data: pd.DataFrame):
            '''
            This is to put the information within the PostProcess class into this sub-class.

            This will be automatically run when an instance of PostProcess is defined.
            '''
            self.demo_data = demo_data
            self.time_data = time_data

        def AgeRegion(self, sample_size, time_sample,
                      comparison=True, sample_strategy='Random',
                      gen_plot: bool = False, saving_path_sampling=None, num_age_group=17,
                      age_group_width=5, seed=None, saving_path_compare=None,
                      scale_method='proportional'):
            '''
            This class is to sample and plot figures using both age and region stratification.

            Parameters:
            -----------

            sample_size : int
                The size of sample
            time_sample : list
                A list of time points to sample the population
            non_responder : bool
                Turn on or off the non-responder function

                Default = False
            non_resp_rate : float between 0 and 1
                The probability that a person does not respond

                Default = None
            comparison : bool
                Turn on or off the comparison between the sampled result and the true result

                Default = True
            sample_strategy : str
                A specific string indicating whether want to change sampled people
                between each sampling

                Strings can be identified: ['Random', 'Same']

                Default = 'Random'
            gen_plot : bool
                Whether or not to generate plots

                Default = False
            saving_path_sampling : str
                The path to save figure showing predicted infection level

                Default = None
            saving_path_compare : str
                The path to save figure showing comparison between predicted
                and true infection level

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
            seed : int or None
                The seed for random numbers

                Default = None

            '''
            res, diff = self._wrapper_Samplers(
                sampling_method='AgeRegion',
                sample_size=sample_size,
                time_sample=time_sample,
                comparison=comparison,
                sample_strategy=sample_strategy,
                gen_plot=gen_plot,
                saving_path_sampling=saving_path_sampling,
                num_age_group=num_age_group,
                age_group_width=age_group_width,
                seed=seed,
                saving_path_compare=saving_path_compare,
                scale_method=scale_method
            )
            return res, diff

        def Region(self, sample_size, time_sample,
                   comparison=True, sample_strategy='Random',
                   gen_plot: bool = False, saving_path_sampling=None,
                   seed=None, saving_path_compare=None,
                   scale_method='proportional'):
            '''
            This class is to sample and plot figures using both age and region stratification.

            Parameters:
            -----------

            sample_size : int
                The size of sample
            time_sample : list
                A list of time points to sample the population
            non_responder : bool
                Turn on or off the non-responder function

                Default = False
            non_resp_rate : float between 0 and 1
                The probability that a person does not respond

                Default = None
            comparison : bool
                Turn on or off the comparison between the sampled result and the true result

                Default = True
            sample_strategy : str
                A specific string indicating whether want to change sampled people
                between each sampling

                Strings can be identified: ['Random', 'Same']

                Default = 'Random'
            gen_plot : bool
                Whether or not to generate plots

                Default = False
            saving_path_sampling : str
                The path to save figure showing predicted infection level

                Default = None
            saving_path_compare : str
                The path to save figure showing comparison between predicted
                and true infection level

                Default = None
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
            seed : int or None
                The seed for random numbers

                Default = None

            '''
            res, diff = self._wrapper_Samplers(
                sampling_method='Region',
                sample_size=sample_size,
                time_sample=time_sample,
                comparison=comparison,
                sample_strategy=sample_strategy,
                gen_plot=gen_plot,
                saving_path_sampling=saving_path_sampling,
                seed=seed,
                saving_path_compare=saving_path_compare,
                scale_method=scale_method,
                num_age_group=17,
                age_group_width=5
            )
            return res, diff

        def Age(self, sample_size, time_sample,
                comparison=True, sample_strategy='Random',
                gen_plot: bool = False, saving_path_sampling=None, num_age_group=17,
                age_group_width=5,
                seed=None, saving_path_compare=None,
                scale_method='proportional'):
            '''
            This class is to sample and plot figures using both age and region stratification.

            Parameters:
            -----------

            sample_size : int
                The size of sample
            time_sample : list
                A list of time points to sample the population
            comparison : bool
                Turn on or off the comparison between the sampled result and the true result

                Default = True
            sample_strategy : str
                A specific string indicating whether want to change sampled people
                between each sampling

                Strings can be identified: ['Random', 'Same']

                Default = 'Random'
            gen_plot : bool
                Whether or not to generate plots

                Default = False
            saving_path_sampling : str
                The path to save figure showing predicted infection level

                Default = None
            saving_path_compare : str
                The path to save figure showing comparison between predicted
                and true infection level

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
            data_store_path : str
                The path to store data generated during sampling

                Default = ./input/
            seed : int or None
                The seed for random numbers

                Default = None

            '''
            res, diff = self._wrapper_Samplers(
                sampling_method='Age',
                sample_size=sample_size,
                time_sample=time_sample,
                comparison=comparison,
                sample_strategy=sample_strategy,
                gen_plot=gen_plot,
                saving_path_sampling=saving_path_sampling,
                num_age_group=num_age_group,
                age_group_width=age_group_width,
                seed=seed,
                saving_path_compare=saving_path_compare,
                scale_method=scale_method
            )
            return res, diff

        def Base(self, sample_size, time_sample,
                 comparison=True, sample_strategy='Random',
                 gen_plot: bool = False, saving_path_sampling=None,
                 seed=None, saving_path_compare=None,
                 scale_method='proportional'):
            '''
            This class is to sample and plot figures using both age and region stratification.

            Parameters:
            -----------

            sample_size : int
                The size of sample
            time_sample : list
                A list of time points to sample the population
            comparison : bool
                Turn on or off the comparison between the sampled result and the true result

                Default = True
            sample_strategy : str
                A specific string indicating whether want to change sampled people
                between each sampling

                Strings can be identified: ['Random', 'Same']

                Default = 'Random'
            gen_plot : bool
                Whether or not to generate plots

                Default = False
            saving_path_sampling : str
                The path to save figure showing predicted infection level

                Default = None
            saving_path_compare : str
                The path to save figure showing comparison between predicted
                and true infection level

                Default = None
            scale_method : str
                Specific string telling how to compare the sampled data with the true population

                Default = 'proportional'
            data_store_path : str
                The path to store data generated during sampling

                Default = ./input/
            seed : int or None
                The seed for random numbers

                Default = None

            '''
            res, diff = self._wrapper_Samplers(
                sampling_method='Base',
                sample_size=sample_size,
                time_sample=time_sample,
                comparison=comparison,
                sample_strategy=sample_strategy,
                gen_plot=gen_plot,
                saving_path_sampling=saving_path_sampling,
                seed=seed,
                saving_path_compare=saving_path_compare,
                scale_method=scale_method,
                num_age_group=17,
                age_group_width=5
            )
            return res, diff

        def _compare(self, time_sample, gen_plot=False, scale_method: str = 'proportional', saving_path_compare=None):
            '''
            Generate a graph comparing the difference between predicted and real infection level
            This method should not be used directly, it is integrated within methods AgeRegion, Age,
            Region and Base.

            '''
            # Based on the input, use different scale method to estimate the true infection number
            if scale_method == 'proportional':
                result_scaled = np.round(np.array(self.result) * len(self.demo_data))

            # Get the true result from self.time_data
            true_result_plot = []
            for t in range(max(time_sample) + 1):
                num = self.time_data.iloc[t, 1:].isin([3, 4, 5, 6, 7, 8]).sum()
                true_result_plot.append(num)
            
            true_result = []
            for t in time_sample:
                true_result.append(true_result_plot[t])

            # Find the difference between estimated infection level and the real one
            diff = np.array(true_result) - result_scaled
            if gen_plot:
                plt.figure()
                plt.plot(time_sample, result_scaled, label='Predicted result', linestyle='--')
                plt.plot(range(max(time_sample) + 1), true_result_plot, label='True result')
                plt.plot(time_sample, np.abs(diff), label='Absolute difference')
                plt.legend()
                plt.xlabel('Time')
                plt.ylabel('Population')
                plt.xlim(0, max(time_sample))
                plt.ylim(0, len(self.demo_data))
                plt.title('Number of infection in the population')
                if saving_path_compare:
                    plt.savefig(saving_path_compare)
            return diff

        def _wrapper_Samplers(self, sampling_method, sample_size, time_sample,
                              comparison=True, sample_strategy='Random',
                              gen_plot: bool = False, saving_path_sampling=None, num_age_group=17,
                              age_group_width=5,
                              seed=None, saving_path_compare=None,
                              scale_method='proportional'):
            '''
            This is the function really doing work.

            The reason why this wrapper function is set up is to reduce repeated code.

            The Age and Base methods share very similar code structures.
            '''
            if seed is not None:
                np.random.seed(seed)
            predicted_total_age = [[] for _ in range(num_age_group)]
            last_row = self.demo_data.iloc[-1]
            region_id = int(last_row['id'].split('.')[0])
            infected_proportion_region = [[] for _ in range(region_id + 1)]

            if sample_strategy == 'Same':  # Do not change people sampled at each sample time point
                infected_rate = []
                peoples = []
                regions = []

                # Do the sampling
                if sampling_method == 'Age':
                    sampler_class = AgeFastSampler(data=self.demo_data, num_age_group=num_age_group,
                                                   age_group_width=age_group_width)
                elif sampling_method == 'Region':
                    sampler_class = RegionFastSampler(data=self.demo_data)
                elif sampling_method == 'AgeRegion':
                    sampler_class = AgeRegionFastSampler(data=self.demo_data, num_age_group=num_age_group,
                                                         age_group_width=age_group_width)
                else:
                    sampler_class = BaseFastSampler(data=self.demo_data)
                people = sampler_class.sample(sample_size=sample_size)

                # Get results of each people sampled
                X = SamplingMaker(non_resp_rate=0, keep_track=True, data=self.time_data,
                                  false_positive=0, false_negative=0, threshold=None)
                ite = X(time_sample, people)

                infected_proportion_region = self.get_infections_by_region(ite, infected_proportion_region,
                                                                               sample_strategy, time_sample)

                # For the IDs, locate their age from the data
                predicted_total_age = self.get_infection_by_groups(people, ite, num_age_group, age_group_width,
                                                                    predicted_total_age, sample_strategy, time_sample)

                # Output the infected rate
                for i in range(len(time_sample)):
                    infected_rate.append(ite.iloc[i].value_counts().get('Positive', 0) / len(people))
            elif sample_strategy == 'Random':  # Change people sampled at each sample time point
                infected_rate = []
                peoples = []
                regions = []
                for i in range(len(time_sample)):  # Sample at each sample time points
                    if sampling_method == 'Age':
                        sampler_class = AgeFastSampler(data=self.demo_data, num_age_group=num_age_group,
                                                       age_group_width=age_group_width)
                    elif sampling_method == 'Region':
                        sampler_class = RegionFastSampler(data=self.demo_data)
                    elif sampling_method == 'AgeRegion':
                        sampler_class = AgeRegionFastSampler(data=self.demo_data, num_age_group=num_age_group,
                                                             age_group_width=age_group_width)
                    else:
                        sampler_class = BaseFastSampler(data=self.demo_data)
                    people = sampler_class.sample(sample_size=sample_size)

                    # Get the results of each people sampled
                    X = SamplingMaker(non_resp_rate=0, keep_track=True, data=self.time_data,
                                      false_positive=0, false_negative=0, threshold=None)
                    ite = X([time_sample[i]], people)

                    infected_proportion_region, regions_sampled = self.get_infections_by_region(ite, infected_proportion_region,
                                                                                                sample_strategy, time_sample)

                    # For the IDs, locate their age from the data
                    predicted_total_age, ages_sampled = self.get_infection_by_groups(people, ite, num_age_group, age_group_width,
                                                                                     predicted_total_age, sample_strategy, time_sample)

                    # Output the infected rate
                    infected_rate.append(ite.iloc[0].value_counts().get('Positive', 0) / len(people))

                    peoples.append(ages_sampled)
                    self.people_sampled = peoples
                    regions.append(regions_sampled)
                    self.region_sampled = regions

            # Plot the figure
            if gen_plot:
                plt.figure()
                infected_population = np.round(np.array(infected_rate) * len(self.demo_data))
                plt.plot(time_sample, infected_population)
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
            self.result_ages = predicted_total_age
            self.result_regions = infected_proportion_region

            if comparison:
                diff = self._compare(time_sample=time_sample, gen_plot=gen_plot, scale_method=scale_method,
                                     saving_path_compare=saving_path_compare)
                return res, diff
            else:
                return res, None
    
        def get_infection_by_groups(self, people, ite, num_age_group, age_group_width, predicted_total_age, sample_strategy, time_sample):
            """Method to get the proportion of infected individuals in each age group.

            Parameters
            ----------
            people: list
                A list of people's IDs
            ite: pandas.DataFrame
                The result of the people sampled
            num_age_group: int
                How many age groups are there
            age_group_width: int
                The width of each age group
            predicted_total_age: list[list, ...]
                A list containing num_age_group lists
            sample_strategy: str
                A specific string indicating whether want to change sampled people
                between each sampling
            time_sample: list
                A list of time points to sample the population

            Returns
            -------
            predicted_total_age: list[list, ...]
                A list of lists, each list contains the infection proportion of each age group at each time step

            """
            ite_age = []
            ages_sampled = [0] * num_age_group
            demo_data_dict = self.demo_data.set_index('id')['age'].to_dict()
            for id in people:
                age_value = demo_data_dict.get(id, 0)
                age_pos = min(num_age_group - 1, math.floor(age_value / age_group_width))
                ite_age.append(age_pos)
                ages_sampled[age_pos] += 1
            ite_age = np.array([[person, age] for person, age in zip(people, ite_age)])
            ite_age = pd.DataFrame(ite_age, columns=['id', 'age'])

            # For the age group within age coloumn of ite_age, use value_counts to get the number of people postive in each age group
            for a in range(num_age_group):
                ite_age_group = ite_age[ite_age['age'] == f'{a}']['id']
                # For this series, use value_counts to get the number of people postive in each age group from ite
                ite_age_group_results = ite[ite_age_group]
                if sample_strategy == 'Random':
                    if ite_age_group_results.empty:
                        predicted_total_age[a].append(0.0)
                    else:
                        if len(ite_age_group) != 0:
                            infected_rate_age_group = ite_age_group_results.iloc[0].value_counts().get('Positive', 0) / len(ite_age_group)
                            predicted_total_age[a].append(infected_rate_age_group)
                elif sample_strategy == 'Same':
                    for i in range(len(time_sample)):
                        if ite_age_group_results.empty:
                            predicted_total_age[a].append(0.0)
                        else:
                            infected_rate_age_group = ite_age_group_results.iloc[i].value_counts().get('Positive', 0) / len(ite_age_group_results.columns)
                            predicted_total_age[a].append(infected_rate_age_group)

            return predicted_total_age, ages_sampled

        def get_infections_by_region(self, ite, infected_proportion_region, sample_strategy, time_sample):
            """Method to get the proportion of infected individuals in each region.

            Parameters
            ----------
            ite: pandas.DataFrame
                The result of the people sampled
            infected_proportion_region: list[list, ...]
                A list containing num_cells empty lists
            sample_strategy: str
                A specific string indicating whether want to change sampled people
                between each sampling
            time_sample: list
                A list of time points to sample the population

            Returns
            -------
            infected_proportion_region: list[list, ...]
                A list of lists, each list contains the infection proportion of each region at each time step
            regions_sampled: list
                A list of integers, each integer represents the number of people sampled in each region

            """
            # Extract region numbers from ids
            region_ids = ite.columns.str.split('.').str[0].astype(int)
            
            # Create a DataFrame with the regions as the index
            ite_new = ite.copy()
            ite_new.columns = region_ids
            
            regions_sampled = [0] * len(infected_proportion_region)

            # Group by the region ids
            grouped = ite_new.T.groupby(by=ite_new.columns)

            for r in range(len(infected_proportion_region)):
                if r in grouped.groups:
                    # Select only columns corresponding to region r
                    ite_cells = grouped.get_group(r)

                    # Update the region sampled count
                    regions_sampled[r] += ite_cells.shape[1]

                    if sample_strategy == 'Random':
                        positive_counts = ite_cells.iloc[0].value_counts().get('Positive', 0)
                        infected_proportion_region[r].append(positive_counts / ite_cells.shape[1])

                    elif sample_strategy == 'Same':
                        for i in range(len(time_sample)):
                            positive_counts = ite_cells.iloc[i].value_counts().get('Positive', 0)
                            infected_proportion_region[r].append(positive_counts / ite_cells.shape[1])
                else:
                    regions_sampled[r] += 0
                    # If no columns for the region, add 0.0 to infected_proportion_region
                    infected_proportion_region[r].append(0.0)

            return infected_proportion_region, regions_sampled
