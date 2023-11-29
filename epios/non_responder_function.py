import numpy as np
import math


def additional_nonresponder(data, nonRespID: list, num_region_group: int, num_age_group: int = 17,
                            sampling_percentage=0.1, proportion=0.01, threshold=None):
    '''
    Generate the additional samples according to the non-response rate
    --------
    Input:
    data(pandas.DataFrame): The population information, corresponding to 'data.csv' in
                            the data pre-process part
    nonRespID(list): A list containing the non-responder IDs
    num_region_group(int): The number of region groups
    num_age_group(int): The number of age groups
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
    df = data
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
