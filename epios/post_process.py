import numpy as np
import matplotlib.pyplot as plt
import math
from epios.sampler import Sampler
from epios.sampler_age_region import SamplerAgeRegion
from epios.sampler_age import SamplerAge
from epios.sampler_region import SamplerRegion
from epios.sampling_maker import SamplingMaker


class PostProcess():

    def __init__(self, demo_data, time_data, data_store_path='./input/'):
        '''
        This is the setup for the post process part of the sampled data
        --------
        Input:
        time_sample(list of int): contains the which time should we sample the data
        sample_strategy(str): should be either 'Random' or 'Same'
                              'random' means sampling random people each round
                              'same' means sampling same people each round

        '''
        self.demo_data = demo_data
        self.time_data = time_data
        self.path = data_store_path

    def __call__(self, sampling_method, sample_size, time_sample, non_responder=False, comparison=True,
                 nonresprate=None, **kwargs):
        if non_responder:
            if nonresprate is None:
                raise ValueError('You have to input the non-response rate when considering non-responders')
            sampling_params = ['gen_plot', 'saving_path', 'num_age_group', 'age_group_width',
                               'sampling_percentage', 'proportion', 'threshold']
            compare_params = ['scale_method', 'saving_path']
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
            res = self.sampled_non_responder(sampling_method=sampling_method, sample_size=sample_size,
                                             time_sample=time_sample, nonresprate=nonresprate, **sampling_input)
            if comparison:
                compare_input = {}
                for i in compare_params:
                    try:
                        compare_input[i] = kwargs[i]
                    except KeyError:
                        pass
                diff = self.compare(time_sample=time_sample, **compare_input)
                return res, diff
            return res
        else:
            sampling_params = ['sample_strategy', 'gen_plot', 'saving_path', 'num_age_group',
                               'age_group_width']
            compare_params = ['scale_method', 'saving_path']
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
            res = self.sampled_result(sampling_method=sampling_method, sample_size=sample_size,
                                      time_sample=time_sample, **sampling_input)
            if comparison:
                compare_input = {}
                for i in compare_params:
                    try:
                        compare_input[i] = kwargs[i]
                    except KeyError:
                        pass
                diff = self.compare(time_sample=time_sample, **compare_input)
                return res, diff
            return res

    def sampled_result(self, sampling_method, sample_size, time_sample, sample_strategy='Random',
                       gen_plot: bool = False, saving_path=None, num_age_group=17, age_group_width=5):
        '''
        This is a method to generate the sampled result and plot a figure
        --------
        Input:
        gen_plot(bool): whether generate a plot or not

        Output:
        res(pd.DataFrame): contains the result of sampling

        '''
        if sampling_method == 'AgeRegion':
            if sample_strategy == 'Same':
                infected_rate = []
                sampler_class = SamplerAgeRegion(data=self.demo_data, data_store_path=self.path,
                                                 num_age_group=num_age_group, age_group_width=age_group_width)
                people = sampler_class.sample(sample_size=sample_size)
                X = SamplingMaker(nonresprate=0, keeptrack=True, TheData=self.time_data,
                                  false_positive=0, false_negative=0, threshold=None)
                ite = X(time_sample, people)
                for i in range(len(time_sample)):
                    infected_rate.append(ite.iloc[i].value_counts().get('Positive', 0) / len(people))
            elif sample_strategy == 'Random':
                infected_rate = []
                for i in range(len(time_sample)):
                    if i == 0:
                        sampler_class = SamplerAgeRegion(data=self.demo_data, data_store_path=self.path,
                                                         num_age_group=num_age_group, age_group_width=age_group_width)
                    else:
                        sampler_class = SamplerAgeRegion(data_store_path=self.path, pre_process=False,
                                                         num_age_group=num_age_group, age_group_width=age_group_width)
                    people = sampler_class.sample(sample_size=sample_size)
                    X = SamplingMaker(nonresprate=0, keeptrack=True, TheData=self.time_data,
                                      false_positive=0, false_negative=0, threshold=None)
                    ite = X([time_sample[i]], people)
                    infected_rate.append(ite.iloc[0].value_counts().get('Positive', 0) / len(people))
        elif sampling_method == 'Age':
            if sample_strategy == 'Same':
                infected_rate = []
                sampler_class = SamplerAge(data=self.demo_data, data_store_path=self.path,
                                           num_age_group=num_age_group, age_group_width=age_group_width)
                people = sampler_class.sample(sample_size=sample_size)
                X = SamplingMaker(nonresprate=0, keeptrack=True, TheData=self.time_data,
                                  false_positive=0, false_negative=0, threshold=None)
                ite = X(time_sample, people)
                for i in range(len(time_sample)):
                    infected_rate.append(ite.iloc[i].value_counts().get('Positive', 0) / len(people))
            elif sample_strategy == 'Random':
                infected_rate = []
                for i in range(len(time_sample)):
                    if i == 0:
                        sampler_class = SamplerAge(data=self.demo_data, data_store_path=self.path,
                                                   num_age_group=num_age_group, age_group_width=age_group_width)
                    else:
                        sampler_class = SamplerAge(data_store_path=self.path, pre_process=False,
                                                   num_age_group=num_age_group, age_group_width=age_group_width)
                    people = sampler_class.sample(sample_size=sample_size)
                    X = SamplingMaker(nonresprate=0, keeptrack=True, TheData=self.time_data,
                                      false_positive=0, false_negative=0, threshold=None)
                    ite = X([time_sample[i]], people)
                    infected_rate.append(ite.iloc[0].value_counts().get('Positive', 0) / len(people))
        elif sampling_method == 'Region':
            if sample_strategy == 'Same':
                infected_rate = []
                sampler_class = SamplerRegion(data=self.demo_data, data_store_path=self.path)
                people = sampler_class.sample(sample_size=sample_size)
                X = SamplingMaker(nonresprate=0, keeptrack=True, TheData=self.time_data,
                                  false_positive=0, false_negative=0, threshold=None)
                ite = X(time_sample, people)
                for i in range(len(time_sample)):
                    infected_rate.append(ite.iloc[i].value_counts().get('Positive', 0) / len(people))
            elif sample_strategy == 'Random':
                infected_rate = []
                for i in range(len(time_sample)):
                    if i == 0:
                        sampler_class = SamplerAgeRegion(data=self.demo_data, data_store_path=self.path)
                    else:
                        sampler_class = SamplerAgeRegion(data_store_path=self.path, pre_process=False)
                    people = sampler_class.sample(sample_size=sample_size)
                    X = SamplingMaker(nonresprate=0, keeptrack=True, TheData=self.time_data,
                                      false_positive=0, false_negative=0, threshold=None)
                    ite = X([time_sample[i]], people)
                    infected_rate.append(ite.iloc[0].value_counts().get('Positive', 0) / len(people))
        elif sampling_method == 'Base':
            if sample_strategy == 'Same':
                infected_rate = []
                sampler_class = Sampler(data=self.demo_data, data_store_path=self.path)
                people = sampler_class.sample(sample_size=sample_size)
                X = SamplingMaker(nonresprate=0, keeptrack=True, TheData=self.time_data,
                                  false_positive=0, false_negative=0, threshold=None)
                ite = X(time_sample, people)
                for i in range(len(time_sample)):
                    infected_rate.append(ite.iloc[i].value_counts().get('Positive', 0) / len(people))
            elif sample_strategy == 'Random':
                infected_rate = []
                for i in range(len(time_sample)):
                    if i == 0:
                        sampler_class = Sampler(data=self.demo_data, data_store_path=self.path)
                    else:
                        sampler_class = Sampler(data_store_path=self.path, pre_process=False)
                    people = sampler_class.sample(sample_size=sample_size)
                    X = SamplingMaker(nonresprate=0, keeptrack=True, TheData=self.time_data,
                                      false_positive=0, false_negative=0, threshold=None)
                    ite = X([time_sample[i]], people)
                    infected_rate.append(ite.iloc[0].value_counts().get('Positive', 0) / len(people))
        else:
            raise ValueError('You must input a valid sampling method')
        if gen_plot:
            plt.plot(time_sample, infected_rate)
            plt.xlabel('Time')
            plt.ylabel('Population')
            plt.xlim(0, max(time_sample))
            plt.ylim(0, len(self.demo_data))
            plt.title('Number of infection in the sample')
            if saving_path:
                plt.savefig(saving_path + 'sample.png')
        res = []
        res.append(time_sample)
        res.append(infected_rate)
        self.result = infected_rate
        return res

    def sampled_non_responder(self, sampling_method, sample_size, time_sample, nonresprate,
                              gen_plot: bool = False, saving_path=None, sampling_percentage=0.1,
                              proportion=0.01, threshold=None, num_age_group=17, age_group_width=5):
        if sampling_method == 'AgeRegion':
            infected_rate = []
            for i in range(len(time_sample)):
                if i == 0:
                    sampler_class = SamplerAgeRegion(data=self.demo_data, data_store_path=self.path,
                                                     num_age_group=num_age_group, age_group_width=age_group_width)
                else:
                    sampler_class = SamplerAgeRegion(data_store_path=self.path, pre_process=False,
                                                     num_age_group=num_age_group, age_group_width=age_group_width)
                try:
                    people = sampler_class.sample(sample_size=sample_size, additional_sample=additional_sample)
                except NameError:
                    people = sampler_class.sample(sample_size=sample_size)
                X = SamplingMaker(nonresprate=nonresprate, keeptrack=True, TheData=self.time_data,
                                  false_positive=0, false_negative=0, threshold=None)
                ite = X([time_sample[i]], people)
                try:
                    additional_sample = np.array(additional_sample)
                    if additional_sample.sum() == 0:
                        raise NameError
                    else:
                        indices = np.nonzero(additional_sample)
                        add_pos = []
                        for k in range(len(indices[0])):
                            add_pos.append((indices[0][k], indices[1][k]))
                        count_total = 0
                        count_posi = 0
                        other_posi = 0
                        count_nonResp = 0
                        other_nonResp = 0
                        for id in people:
                            region_pos = int(id.split('.')[0])
                            age_value = self.demo_data[self.demo_data['ID'] == id]['age'].values[0]
                            age_pos = min(num_age_group - 1, math.floor(age_value / age_group_width))
                            indexer = (region_pos, age_pos)
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

                nonRespID = []
                for j in range(len(ite.columns)):
                    if ite.iloc[0, j] == 'NonResponder':
                        nonRespID.append(ite.columns[j])
                additional_sample = sampler_class.additional_nonresponder(nonRespID=nonRespID,
                                                                          sampling_percentage=sampling_percentage,
                                                                          proportion=proportion, threshold=threshold)
        elif sampling_method == 'Age':
            raise ValueError('Age stratification method does not support non-responders, please disable non-responders')
        elif sampling_method == 'Region':
            infected_rate = []
            for i in range(len(time_sample)):
                if i == 0:
                    sampler_class = SamplerRegion(data=self.demo_data, data_store_path=self.path)
                else:
                    sampler_class = SamplerRegion(data_store_path=self.path, pre_process=False)
                try:
                    people = sampler_class.sample(sample_size=sample_size, additional_sample=additional_sample)
                except NameError:
                    people = sampler_class.sample(sample_size=sample_size)
                X = SamplingMaker(nonresprate=nonresprate, keeptrack=True, TheData=self.time_data,
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

                nonRespID = []
                for j in range(len(ite.columns)):
                    if ite.iloc[0, j] == 'NonResponder':
                        nonRespID.append(ite.columns[j])
                additional_sample = sampler_class.additional_nonresponder(nonRespID=nonRespID,
                                                                          sampling_percentage=sampling_percentage,
                                                                          proportion=proportion, threshold=threshold)
        elif sampling_method == 'Base':
            raise ValueError('Random sampling does not support non-responders, please disable non-responders')
        else:
            raise ValueError('You must input a valid sampling method')

        if gen_plot:
            plt.plot(time_sample, infected_rate)
            plt.xlabel('Time')
            plt.ylabel('Proportion of population')
            plt.xlim(0, max(time_sample))
            plt.ylim(0, 1)
            plt.title('Proportion of infection in the sample (consider non-responders)')
            if saving_path:
                plt.savefig(saving_path + 'sample_nonResp.png')
        res = []
        res.append(time_sample)
        res.append(infected_rate)
        self.result = infected_rate
        return res

    def compare(self, time_sample, scale_method: str = 'proportional', saving_path=None):
        '''
        Generate a graph comparing the difference between predicted and real infection level
        -------
        Input:
        scale_method(str): should be a specific string
                           'proportional' means directly enlarge sample result to get an estimate
        saving_path(str): The path to save the figure

        Output:
        diff(numpy.array): an array of difference between the predicted and real infection level

        '''
        if scale_method == 'proportional':
            result_scaled = np.array(self.result) * len(self.demo_data)
        true_result = []
        for t in time_sample:
            num = self.time_data.iloc[t].value_counts().get('InfectASympt', 0)
            num += self.time_data.iloc[t].value_counts().get('InfectMild', 0)
            num += self.time_data.iloc[t].value_counts().get('InfectGP', 0)
            num += self.time_data.iloc[t].value_counts().get('InfectHosp', 0)
            num += self.time_data.iloc[t].value_counts().get('InfectICU', 0)
            num += self.time_data.iloc[t].value_counts().get('InfectICURecov', 0)
            true_result.append(num)
        diff = np.array(true_result) - result_scaled
        plt.plot(time_sample, result_scaled, label='Predicted result')
        plt.plot(time_sample, true_result, label='True result')
        plt.plot(time_sample, diff, label='Difference')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.title('Number of infection in the population')
        if saving_path:
            plt.savefig(saving_path + 'compare.png')
        return diff
