import numpy as np
import matplotlib.pyplot as plt
import math
from epios.sampler_age_region import SamplerAgeRegion
from epios.sampling_maker import SamplingMaker


class PostProcess():

    def __init__(self, demo_data, time_data, sample_size, time_sampled: list,
                 data_store_path='./input/', sample_strategy: str = 'random'):
        '''
        This is the setup for the post process part of the sampled data
        --------
        Input:
        time_sampled(list of int): contains the which time should we sample the data
        sample_strategy(str): should be either 'random' or 'same'
                              'random' means sampling random people each round
                              'same' means sampling same people each round

        '''
        self.time_sample = time_sampled
        self.sample_strategy = sample_strategy
        self.demo_data = demo_data
        self.time_data = time_data
        self.sample_size = sample_size
        self.path = data_store_path

    def sampled_result(self, gen_plot: bool = False, saving_path=None):
        '''
        This is a method to generate the sampled result and plot a figure
        --------
        Input:
        gen_plot(bool): whether generate a plot or not

        Output:
        res(pd.DataFrame): contains the result of sampling

        '''
        if self.sample_strategy == 'same':
            infected_number = []
            sampler_class = SamplerAgeRegion(data=self.demo_data, data_store_path=self.path)
            people = sampler_class.sample(sample_size=self.sample_size)
            X = SamplingMaker(nonresprate=0, keeptrack=True, TheData=self.time_data,
                              false_positive=0, false_negative=0, threshold=None)
            ite = X(self.time_sample, people)
            for i in range(len(self.time_sample)):
                infected_number.append(ite.iloc[i].value_counts().get('Positive', 0))
        elif self.sample_strategy == 'random':
            infected_number = []
            for i in range(len(self.time_sample)):
                if i == 0:
                    sampler_class = SamplerAgeRegion(data=self.demo_data, data_store_path=self.path)
                else:
                    sampler_class = SamplerAgeRegion(data_store_path=self.path, pre_process=False)
                people = sampler_class.sample(sample_size=self.sample_size)
                X = SamplingMaker(nonresprate=0, keeptrack=True, TheData=self.time_data,
                                  false_positive=0, false_negative=0, threshold=None)
                ite = X([self.time_sample[i]], people)
                infected_number.append(ite.iloc[0].value_counts().get('Positive', 0))
        if gen_plot:
            plt.plot(self.time_sample, infected_number)
            plt.xlabel('Time')
            plt.ylabel('Population')
            plt.xlim(0, max(self.time_sample))
            plt.ylim(0, len(self.demo_data))
            plt.title('Number of infection in the sample')
            if saving_path:
                plt.savefig(saving_path + 'sample.png')
        res = []
        res.append(self.time_sample)
        res.append(infected_number)
        self.result = infected_number
        return res

    def sampled_non_responder(self, nonresprate, gen_plot: bool = False, saving_path=None):
        if self.sample_strategy == 'same':
            raise ValueError("non-responders can only be introduced when the strategy is 'random'.")
        elif self.sample_strategy == 'random':
            infected_rate = []
            for i in range(len(self.time_sample)):
                if i == 0:
                    sampler_class = SamplerAgeRegion(data=self.demo_data, data_store_path=self.path)
                else:
                    sampler_class = SamplerAgeRegion(data_store_path=self.path, pre_process=False)
                try:
                    people = sampler_class.sample(sample_size=self.sample_size, additional_sample=additional_sample)
                except NameError:
                    people = sampler_class.sample(sample_size=self.sample_size)
                X = SamplingMaker(nonresprate=nonresprate, keeptrack=True, TheData=self.time_data,
                                  false_positive=0, false_negative=0, threshold=None)
                ite = X([self.time_sample[i]], people)
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
                            age_pos = min(16, math.floor(self.demo_data[self.demo_data['ID'] == id]['age'] / 5))
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
                            spaces = self.sample_size - (len(people) - count_total)
                            spaces_posi = round(spaces * count_posi / effective_total)
                            infected_rate.append((spaces_posi + other_posi)
                                                 / (spaces + len(people) - count_total - other_nonResp))
                        else:
                            infected_rate.append(other_posi / (len(people) - count_total - other_nonResp))
                except NameError:
                    infected_rate.append(ite.iloc[0].value_counts().get('Positive', 0)
                                         / (ite.iloc[0].value_counts().get('Positive', 0)
                                            + ite.iloc[0].value_counts().get('Negative', 0)))

                nonRespID = []
                for j in range(len(ite.columns)):
                    if ite.iloc[0, j] == 'NonResponder':
                        nonRespID.append(ite.columns[j])
                additional_sample = sampler_class.additional_nonresponder(nonRespID=nonRespID)

        if gen_plot:
            plt.plot(self.time_sample, infected_rate)
            plt.xlabel('Time')
            plt.ylabel('Proportion of population')
            plt.xlim(0, max(self.time_sample))
            plt.ylim(0, 1)
            plt.title('Proportion of infection in the sample (consider non-responders)')
            if saving_path:
                plt.savefig(saving_path + 'sample_nonResp.png')
        res = []
        res.append(self.time_sample)
        res.append(infected_rate)
        return res

    def compare(self, scale_method: str = 'proportional', saving_path=None):
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
            scale_para = len(self.demo_data) / self.sample_size
            result_scaled = np.array(self.result) * scale_para
        true_result = []
        for t in self.time_sample:
            num = self.time_data.iloc[t].value_counts().get('InfectASympt', 0)
            num += self.time_data.iloc[t].value_counts().get('InfectMild', 0)
            num += self.time_data.iloc[t].value_counts().get('InfectGP', 0)
            num += self.time_data.iloc[t].value_counts().get('InfectHosp', 0)
            num += self.time_data.iloc[t].value_counts().get('InfectICU', 0)
            num += self.time_data.iloc[t].value_counts().get('InfectICURecov', 0)
            true_result.append(num)
        diff = np.array(true_result) - result_scaled
        plt.plot(self.time_sample, result_scaled, label='Predicted result')
        plt.plot(self.time_sample, true_result, label='True result')
        plt.plot(self.time_sample, diff, label='Difference')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.title('Number of infection in the population')
        if saving_path:
            plt.savefig(saving_path + 'compare.png')
        return diff
