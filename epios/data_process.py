import numpy as np
import pandas as pd
import math
import json


class DataProcess():

    def __init__(self, data: pd.DataFrame, path: str = './input/', num_age_group=17, age_group_width=5, mode=None):
        '''
        .data attribute contains the DataFrame with two columns. The first column
        contains IDs, the second one contains ages

        '''
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
        self.age_group_width = age_group_width
        self.pre_process(path=path, num_age_group=num_age_group)

    def pre_process(self, path='./input/', num_age_group=17):
        '''
        Take the DataFrame then convert the data into files that age_region.py can use
        -------
        Input:
        path(str): The path to save the processed data file
        num_age_group(int): How many age group want to have. Each age group has width 5

        Output:
        Will write three files into the given path
        The first one is data.csv, contains the data for each person
        The second one is microcells.csv, contains the geographical information
        The third one is pop_dist.json, contains a list of age distribution across the population

        '''
        df = self.data
        if self.gen_ageinfo and self.gen_geoinfo:
            population_info = pd.DataFrame(columns=['ID', 'age', 'cell', 'microcell', 'household'])
            household_info = {}
            population_size = len(df)
            count_age = [0] * num_age_group
            for index, row in df.iterrows():
                ind_age = math.floor(row['age'] / self.age_group_width)
                if ind_age < num_age_group - 1:
                    count_age[ind_age] += 1
                else:
                    count_age[-1] += 1
                person_id = row['ID']
                splitted_id = person_id.split('.')
                cell_num = int(splitted_id[0])
                microcell_num = int(splitted_id[1])
                household_num = int(splitted_id[2])
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
                new_row = pd.DataFrame({'cell': cell_num, 'microcell': microcell_num,
                                        'household': household_num, 'Susceptible': value}, index=[0])
                household_df = pd.concat([household_df, new_row], ignore_index=True)
            household_df.to_csv(path + 'microcells.csv', index=False)

            age_dist = list(np.array(count_age) / population_size)
            json_string = json.dumps(age_dist)
            with open(path + 'pop_dist.json', 'w') as f:
                f.write(json_string)
        elif self.gen_ageinfo and (~self.gen_geoinfo):
            df.to_csv(path + 'data.csv', index=False)
            population_size = len(df)
            count_age = [0] * num_age_group
            for index, row in df.iterrows():
                ind_age = math.floor(row['age'] / self.age_group_width)
                if ind_age < num_age_group - 1:
                    count_age[ind_age] += 1
                else:
                    count_age[-1] += 1
            age_dist = list(np.array(count_age) / population_size)
            json_string = json.dumps(age_dist)
            with open(path + 'pop_dist.json', 'w') as f:
                f.write(json_string)
        elif self.gen_geoinfo and (~self.gen_ageinfo):
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
            df.to_csv(path + 'data.csv', index=False)
