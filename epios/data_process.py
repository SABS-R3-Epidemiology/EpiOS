import numpy as np
import pandas as pd
import math
import json


class DataProcess():
    """"
    Can process data depending on different modes of samplers.
    When defining an instance, the pre_process part would automatically run.
    This is the base class for different samplers.

    Parameters
    ----------

    path : str
        The path to store the processed data
    data : pandas.DataFrame
        The dataframe containing geographical data
    num_age_group : int
        This will be used when age stratification is enabled,
        indicating how many age groups are there.
        *The last group includes age >= some threshold
    age_group_width : int
        This will beused when age stratification is enabled
        indicating the width of each age group(except for the last group)
    mode : str
        This indicates the specific mode to process the data
        This should be the name of the modes that can be identified

    Attributes
    ----------

    gen_ageinfo : bool
        Whether generating age information
    gen_geoinfo : bool
        Whether generating demographical information
    data : pandas.DataFrame
        The demographical data from EpiABM
    """

    def __init__(self, data: pd.DataFrame, path: str = './input/', num_age_group=None, age_group_width=None, mode=None):
        self.gen_ageinfo = False
        self.gen_geoinfo = False
        if mode == 'AgeRegion':
            self.gen_ageinfo = True
            self.gen_geoinfo = True
        elif mode == 'Base':
            self.gen_ageinfo = False
            self.gen_ageinfo = False
        elif mode == 'Age':
            self.gen_ageinfo = True
            self.gen_geoinfo = False
        elif mode == 'Region':
            self.gen_ageinfo = False
            self.gen_geoinfo = True
        self.data = data
        self.pre_process(path=path, num_age_group=num_age_group, age_group_width=age_group_width)

    def pre_process(self, path='./input/', num_age_group=None, age_group_width=None):
        """
        Take the geographical DataFrame then convert the data into files that Sampler classes can use

        Parameters
        ----------

        (See explanation in __init__ method)

        Output
        ------

        Will write three files(depending on the mode of processing chosen) into the given path
        The first one is data.csv, contains the data for each person
        The second one is microcells.csv, contains the geographical information
        The third one is pop_dist.json, contains a list of age distribution across the population
        """
        df = self.data
        if self.gen_ageinfo and self.gen_geoinfo:
            # Both age and region stratification is needed
            population_info = pd.DataFrame(columns=['ID', 'age', 'cell', 'microcell', 'household'])
            household_info = {}
            population_size = len(df)
            count_age = [0] * num_age_group
            for index, row in df.iterrows():
                ind_age = math.floor(row['age'] / age_group_width)
                if ind_age < num_age_group - 1:
                    count_age[ind_age] += 1
                else:
                    count_age[-1] += 1
                person_id = row['ID']
                splitted_id = person_id.split('.')
                cell_num = int(splitted_id[0])
                microcell_num = int(splitted_id[1])
                household_num = int(splitted_id[2])
                # Generation of each row of data.csv file
                new_row = pd.DataFrame({'ID': person_id, 'age': row['age'], 'cell': cell_num,
                                        'microcell': microcell_num, 'household': household_num}, index=[0])
                population_info = pd.concat([population_info, new_row], ignore_index=True)
                key = '.'.join(splitted_id[:-1])
                try:
                    household_info[key] += 1
                except KeyError:
                    household_info[key] = 1
            population_info.to_csv(path + 'data.csv', index=False)

            household_df = pd.DataFrame(columns=['cell', 'microcell', 'household', 'Susceptible'])
            for key, value in household_info.items():
                pos_dot = []
                for i in range(len(key)):
                    if key[i] == '.':
                        pos_dot.append(i)
                cell_num = int(key[0:pos_dot[0]])
                microcell_num = int(key[pos_dot[0] + 1:pos_dot[1]])
                household_num = int(key[pos_dot[1] + 1:])
                # Generation of each row of microcells.csv file
                new_row = pd.DataFrame({'cell': cell_num, 'microcell': microcell_num,
                                        'household': household_num, 'Susceptible': value}, index=[0])
                household_df = pd.concat([household_df, new_row], ignore_index=True)
            household_df.to_csv(path + 'microcells.csv', index=False)

            # Generation of pop_dist.json file
            age_dist = list(np.array(count_age) / population_size)
            json_string = json.dumps(age_dist)
            with open(path + 'pop_dist.json', 'w') as f:
                f.write(json_string)
        elif self.gen_ageinfo and (~self.gen_geoinfo):
            # Only age stratification needed
            df.to_csv(path + 'data.csv', index=False)
            population_size = len(df)
            count_age = [0] * num_age_group
            for index, row in df.iterrows():
                ind_age = math.floor(row['age'] / age_group_width)
                if ind_age < num_age_group - 1:
                    count_age[ind_age] += 1
                else:
                    count_age[-1] += 1
            age_dist = list(np.array(count_age) / population_size)
            json_string = json.dumps(age_dist)
            with open(path + 'pop_dist.json', 'w') as f:
                f.write(json_string)
        elif self.gen_geoinfo and (~self.gen_ageinfo):
            # Only region stratification needed
            population_info = pd.DataFrame(columns=['ID', 'cell', 'microcell', 'household'])
            household_info = {}
            population_size = len(df)
            for index, row in df.iterrows():
                person_id = row['ID']
                splitted_id = person_id.split('.')
                cell_num = int(splitted_id[0])
                microcell_num = int(splitted_id[1])
                household_num = int(splitted_id[2])
                new_row = pd.DataFrame({'ID': person_id, 'cell': cell_num,
                                        'microcell': microcell_num, 'household': household_num}, index=[0])
                population_info = pd.concat([population_info, new_row], ignore_index=True)
                key = '.'.join(splitted_id[:-1])
                try:
                    household_info[key] += 1
                except KeyError:
                    household_info[key] = 1
            population_info.to_csv(path + 'data.csv', index=False)

            household_df = pd.DataFrame(columns=['cell', 'microcell', 'household', 'Susceptible'])
            for key, value in household_info.items():
                pos_dot = []
                for i in range(len(key)):
                    if key[i] == '.':
                        pos_dot.append(i)
                cell_num = int(key[0:pos_dot[0]])
                microcell_num = int(key[pos_dot[0] + 1:pos_dot[1]])
                household_num = int(key[pos_dot[1] + 1:])
                new_row = pd.DataFrame({'cell': cell_num, 'microcell': microcell_num,
                                        'household': household_num, 'Susceptible': value}, index=[0])
                household_df = pd.concat([household_df, new_row], ignore_index=True)
            household_df.to_csv(path + 'microcells.csv', index=False)
        elif (~self.gen_geoinfo) and (~self.gen_ageinfo):
            # Neither of age and region stratification needed
            df.to_csv(path + 'data.csv', index=False)
