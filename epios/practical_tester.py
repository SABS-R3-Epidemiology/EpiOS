import numpy as np
import matplotlib.pyplot as plt
from epios import Sampler


class PracticalSampler():

    def __init__(self, time_data, demo_data, non_resp_rate=0, positivity_curve=None):
        self.time_data = time_data
        self.demo_data = demo_data
        self.non_resp_rate = non_resp_rate
        if positivity_curve is None:
            self.positivity_curve = 1
        else:
            if len(positivity_curve) < len(time_data) + 1:
                self.positivity_curve = np.zeros(len(time_data) + 1)
                self.positivity_curve[:len(positivity_curve)] = positivity_curve
            else:
                self.positivity_curve = positivity_curve
    
    def _test_one_person(self, id, time):
        status = self.time_data.loc[time, id]
        if np.random.rand() < self.non_resp_rate:
            return 'Non-Responsive'
        if status <= 1 or status >= 9:
            return 'Negative'
        else:
            infection_history = self.time_data.loc[:time, id].values
            infected_day = np.where(infection_history > 1)[0][0]
            days_infection = time - infected_day
            if isinstance(self.positivity_curve, int):
                threshold = self.positivity_curve
            else:
                threshold = self.positivity_curve[days_infection]
            if np.random.rand() < threshold:
                return 'Positive'
            else:
                return 'Negative'
    
    def test(self, id_list, time):
        results = []
        for id in id_list:
            results.append(self._test_one_person(id, time))
        return results

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
            num = self.time_data.iloc[t, 1:].value_counts().get(3, 0)
            num += self.time_data.iloc[t, 1:].value_counts().get(4, 0)
            num += self.time_data.iloc[t, 1:].value_counts().get(5, 0)
            num += self.time_data.iloc[t, 1:].value_counts().get(6, 0)
            num += self.time_data.iloc[t, 1:].value_counts().get(7, 0)
            num += self.time_data.iloc[t, 1:].value_counts().get(8, 0)
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

    def predict_base(self, sample_size, time_sample, data_store_path='./input/', sample_strategy='Random', seed=None, saving_path_sampling=None, gen_plot=True, comparison=True, scale_method='proportional', saving_path_compare=None):
        if seed is not None:
            np.random.seed(seed)

        if sample_strategy == 'Same':  # Do not change people sampled at each sample time point
            infected_rate = []

            # Do the sampling
            sampler_class = Sampler(data=self.demo_data, data_store_path=data_store_path)
            people = sampler_class.sample(sample_size=sample_size)

            # Get results of each people sampled and output the infected rate
            for i in range(len(time_sample)):
                ite = self.test(people, time_sample[i])
                infected_rate.append(ite.count('Positive') / len(people))
        elif sample_strategy == 'Random':  # Change people sampled at each sample time point
            infected_rate = []
            for i in range(len(time_sample)):  # Sample at each sample time points
                if i == 0:  # First time sampling, need pre_process
                    sampler_class = Sampler(data=self.demo_data, data_store_path=data_store_path)
                else:  # After the data process, we can directly read files processed at the first time
                    sampler_class = Sampler(data=self.demo_data, data_store_path=data_store_path,
                                                pre_process=False)
                people = sampler_class.sample(sample_size=sample_size)

                # Get the results of each people sampled
                ite = self.test(people, time_sample[i])

                # Output the infected rate
                infected_rate.append(ite.count('Positive') / len(people))

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
        self.result = infected_rate
        if comparison:
            diff = self._compare(time_sample=time_sample, gen_plot=gen_plot, scale_method=scale_method,
                                    saving_path_compare=saving_path_compare)
            return res, diff
        else:
            return res, None
