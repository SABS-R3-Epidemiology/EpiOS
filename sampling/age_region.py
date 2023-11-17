import pandas as pd
import numpy as np
import json


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

    def sample(self, sample_size):
        res = []
        df = self.data
        age_dist = self.get_age_dist()
        region_dist = self.get_region_dist()
        ar_dist = np.array(age_dist) * np.array(region_dist).reshape((-1, 1))
        ar_dist = ar_dist.reshape((1, -1))
        size = sample_size
        num_sample = np.random.multinomial(size, ar_dist, size=1)[0]
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
