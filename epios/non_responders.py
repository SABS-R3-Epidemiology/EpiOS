import pandas as pd
import numpy as np
from epios.age_region import Sampler
import math


class NonResponder(Sampler):

    def __init__(self, geoinfo, ageinfo_path, data, nonRespRate):
        super().__init__(geoinfo, ageinfo_path, data)
        self.nonRespRate = nonRespRate

    def additional_sample(self, sampling_percentage=0.1, proportion=0.01, threshold=None):
        df = self.data
        n = len(self.data)
        nonRespRate = self.nonRespRate.copy()
        if threshold is None:
            num_grp = proportion * n
        else:
            num_grp = max(proportion * n, threshold)

        # Assume the nonRespRate is a (1, -1) shape array
        num_age = len(self.get_age_dist())
        num_region = len(self.get_region_dist())
        res = []
        for i in range(num_grp):
            if max(nonRespRate) > 0:
                pos = nonRespRate.index(max(nonRespRate))
                nonRespRate[i] = 0
                res.append([pos % num_age, math.floor(pos / num_age)])
        additional_sample = [[0] * num_age] * num_region
        cap_block = []
        for i in range(len(self.nonRespRate)):
            pos_age = i % num_age
            pos_region = math.floor(i / num_age)
            ite = df[df['cell'] == pos_region]
            if pos_age != num_age - 1:
                ite = ite[ite['age'] >= pos_age * 5]
                ite = ite[ite['age'] < pos_age * 5 + 5]
            else:
                ite = ite[ite['age'] >= pos_age * 5]
            cap_block.append(len(ite))
        cap_block = cap_block.reshape((-1, num_age))
        for i in res:
            additional_sample[i] = round(0.1 * cap_block[i])
        return additional_sample

    def new_idea_postprocessing(self, p, pre_result, symptomatic_profile):
        non_symp = pd.concat([pre_result[pre_result['Status'] == 'S'], pre_result[pre_result['Status'] == 'E'], pre_result[pre_result['Status'] == 'I_asymp']], ignore_index=True)
        t = self.data.time
        non_symp_rate = symptomatic_profile['S'][t] + symptomatic_profile['E'][t] + symptomatic_profile['I_asymp'][t]
        # Assume in symptomatic profile, the values are percentages already
        total_C = len(non_symp) / non_symp_rate
        if total_C < pre_result.num_respond:
            pass
