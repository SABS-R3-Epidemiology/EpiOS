import numpy as np
import matplotlib.pyplot as plt
from sampler_age_region import SamplerAgeRegion
from sampling_maker import SamplingMaker


class PostProcess():

    def __init__(self, demo_data, time_data, sample_size, time_sampled: list, sample_strategy: str = 'random'):
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

    def sampled_result(self, sample_size, gen_plot: bool = False, saving_path:str = './output/'):
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
            sampler_class = SamplerAgeRegion(data=self.demo_data)
            people = sampler_class.sample(sample_size=self.sample_size)
            X = SamplingMaker(nonresprate=0, keeptrack=True, TheData=self.time_data,
                false_positive=0, false_negative=0, threshold=None)
            ite = X(self.time_sample, people)
            for i in range(len(self.time_sample)):
                infected_number.append(ite.iloc[i].value_counts.get('Positive', 0))
        elif self.sample_strategy == 'random':
            infected_number = []
            for i in range(len(self.time_sample)):
                sampler_class = SamplerAgeRegion(data=self.demo_data)
                people = sampler_class.sample(sample_size=self.sample_size)
                X = SamplingMaker(nonresprate=0, keeptrack=False, TheData=self.time_data,
                false_positive=0, false_negative=0, threshold=None)
                ite = X([self.time_sample[i]], people)
                infected_number.append(ite.iloc[0].value_counts.get('Positive', 0))
        if gen_plot:
            plt.plot(self.time_sample, infected_number)
            plt.xlabel('Time')
            plt.ylabel('Population')
            plt.title('Number of infection in the sample')
            plt.savefig(saving_path + 'sample.png')
        res = []
        res.append(self.time_sample)
        res.append(infected_number)
        self.result = infected_number
        return res

    def compare(self, scale_method: str = 'proportional', saving_path: str = './output/'):
        if scale_method == 'proportional':
            scale_para = len(self.demo_data) / self.sample_size
            result_scaled = np.array(self.result) * scale_para
        else:
            pass
        true_result = []
        for t in self.time_sample:
            true_result.append(self.time_data.iloc[t].value_counts.get([2, 3, 4, 5, 6, 7, 8, 9], 0))
        diff = np.array(true_result) - result_scaled
        plt.plot(self.time_sample, result_scaled, label='Predicted result')
        plt.plot(self.time_sample, true_result, label='True result')
        plt.plot(self.time_sample, diff, label='Difference')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.title('Number of infection in the population')
        plt.savefig(saving_path + 'compare.png')
        return diff
