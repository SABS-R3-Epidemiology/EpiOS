import random
import numpy as np

class HouseholdLimits():

    def __init__(self, household_cap, cell, minicell, ids):

        self.household_cap = household_cap
        self.cell = cell
        self.minicell = minicell
        self.ids = ids

    # function to sort samples into a multi-dim list by household
    def sort_samples_by_household(self, samples):

        last_item = samples[len(samples) - 1]
        last_household = int(last_item[4])

        sorted_samples = []
        sample_index = 0
        household_samples = []
        previous_household = 0

        for s in samples:

            current_household = int(s[4])

            if current_household != previous_household:

                previous_household = current_household
                sorted_samples.append(household_samples)
                household_samples = []
            
            household_samples.append(s)

            sample_index += 1
            
        sorted_samples.append(household_samples)
        print(sorted_samples)
            


# Main functions

    # function to get max number of people sampled per household
    def get_max_number(self, h):

        # limit number of people sampled for each household
        if self.household_cap[0] == "cap_number":

            return self.household_cap[1]

        # limit proportion of people sampled across every household
        elif self.household_cap[0] == "cap_proportion":

            return int(self.household_cap[1] * h)

    # function to get the total number of people in each household
    def get_household_totals(self):

        # list containing total no. people in each household
        households = []

        # storage variable
        previous_household = 0

        # store tally of people in each household
        num_people = 0

        # loop through every id
        for i in self.ids:
        
            # household number
            household = int(i[4])
            
            # check if moved on to next household
            if household != previous_household:
            
                # add the tally to the households list
                households.append(num_people)
            
                # re-set tally of people
                num_people = 1
            
                # re-set storage variable
                previous_household = household
            
            else:
            
                # otherwise increment tally
                num_people += 1

        # add total of final household
        households.append(num_people)

        return households

    def get_sample_ids(self):

        # get the total number of people in each household
        households = self.get_household_totals()

        # list of ids to sample from
        id_samples = []

        # initialise household number
        household_num = 0

        # loop over every household
        for h in households:

            # get maximum number of people that can be sampled for each household
            max_num = self.get_max_number(h)

            # check if household has more people than max to sample from
            if h > max_num:

                # select random ids from the hosuehold to sample from
                sample = random.sample(range(0, h), max_num)

            else:
                
                # otherwise sample everyone in that household
                sample = list(range(h))
                    
            # loop through the whole sample
            for s in range(len(sample)):

                # add the string id's of those to be sampled
                id_samples.append(str(self.cell)+"."+str(self.minicell)+"."+str(household_num)+"."+str(sample[s]))

            # increment household number
            household_num += 1

        return id_samples


# Testing

# the current method used for the household limiting sampling
#household_cap = ["cap_number", 3] 
household_cap = ["cap_proportion", 0.5]

# would loop through each cell and minicell
cell = 0
minicell = 0

# dummy data of ids
ids = ["0.0.0.0","0.0.0.1","0.0.0.2",
        "0.0.1.0","0.0.1.1","0.0.1.2","0.0.1.3",
        "0.0.2.0","0.0.2.1","0.0.2.2",
        "0.0.3.0","0.0.3.1","0.0.3.2",
        "0.0.4.0","0.0.4.1","0.0.4.2"]

Test = HouseholdLimits(household_cap, cell, minicell, ids)

sample_ids = Test.get_sample_ids()

#print(sample_ids)

Test.sort_samples_by_household(sample_ids)
