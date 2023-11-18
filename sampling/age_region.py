import pandas as pd
import numpy as np
import json
import math


class Sampler():

    def __init__(self, info, data):
        self.info = info
        self.data = data

    def get_age_dist(self, path):
        with open(path, 'r') as f:
            cofig = json.loads(f.read())
        return cofig

    def get_region_dist(self, path):
        df = pd.read_csv(path)
        n = df['Susceptible'].sum()
        dist = []
        for i in range(df['cell'].max() + 1):
            dist.append(df[df['cell'] == i]['Susceptible'].sum() / n)
        return dist

    def bool_exceed(self, current_age, current_region, cap_age, cap_region):
        if current_age + 2 > cap_age:
            return False
        elif current_region + 2 > cap_region:
            return False
        else:
            return True

    def multinomial_draw(self, n, prob):
        threshold = []
        for i in len(prob):
            try:
                threshold.append(threshold[-1] + prob[i - 1])
            except:
                threshold.append(0)
        prob = prob.reshape((-1, len(self.get_age_dist())))
        cap_age = []
        cap_region = []
        for i in range(np.shape(prob)[1]):
            cap_age.append(n * prob[:, i].sum() + 0.01 * n)
        for i in range(np.shape(prob)[0]):
            cap_region.append(n * prob[i, :].sum() + 0.005 * n)
        prob = prob.reshape((1, -1))
        res = [0] * len(prob)
        current_age = [0] * len(cap_age)
        current_region = [0] * len(cap_region)
        for i in range(n):
            rand = np.random.rand()
            for j in range(len(threshold)):
                if rand > threshold[j]:
                    pos_age = j % len(cap_age)
                    pos_region = math.floor(j / len(cap_age))
                    if self.bool_exceed(current_age[pos_age], current_region[pos_region], cap_age[pos_age], cap_region[pos_region]):
                        res[j] += 1
                        current_age[pos_age] += 1
                        current_region[pos_region] += 1
                    else:
                        res[j] += 1
                        current_age[pos_age] += 1
                        current_region[pos_region] += 1
                        prob = prob.reshape((-1, len(cap_age)))
                        if current_age[pos_age] + 1 > cap_age[pos_age]:
                            prob_exceed = prob[:, pos_age].sum()
                            np.delete(prob, pos_age, 1)
                            prob = prob / (1 - prob_exceed)
                            cap_age.pop(pos_age)
                            threshold = []
                            for k in len(prob):
                                try:
                                    threshold.append(threshold[-1] + prob[i - 1])
                                except:
                                    threshold.append(0)
                            prob = prob.reshape((-1, len(cap_age)))
                        if current_region[pos_region] + 1 > cap_region[pos_region]:
                            prob_exceed = prob[pos_region, :].sum()
                            np.delete(prob, pos_region, 0)
                            prob = prob / (1 - prob_exceed)
                            cap_region.pop(pos_region)
                            threshold = []
                            for k in len(prob):
                                try:
                                    threshold.append(threshold[-1] + prob[i - 1])
                                except:
                                    threshold.append(0)
                            prob = prob.reshape((-1, len(cap_age)))
                    break
        return res

    def sample(self, sample_size):
        res = []
        df = self.data
        age_dist = self.get_age_dist()
        region_dist = self.get_region_dist()
        ar_dist = np.array(age_dist) * np.array(region_dist).reshape((-1, 1))
        ar_dist = ar_dist.reshape((1, -1))
        size = sample_size
        num_sample = self.multinomial_draw(size, ar_dist)
        num_sample = num_sample.reshape((len(region_dist), -1))
        for i in range(len(num_sample)):
            for j in range(len(num_sample[0])):
                if j != len(num_sample) - 1:
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
