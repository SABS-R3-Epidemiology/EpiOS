import random
import numpy as np

class SampleHouseholds():

    def __init__(self, household_cap, ids):

        self.household_cap = household_cap
        self.ids = ids

    def cap_household(self):

        def check_household_cap(household_counter, household_cap, new_sample):

            if household_counter > household_cap:

                    num_samples = household_cap
                    
            else:

                num_samples = household_counter

            households_sample = random.sample(current_households, num_samples)

            for h in households_sample:

                new_sample.append(h)
            
            return new_sample


        # sort ids into order of household, minicell then cell
        self.ids.sort()

        new_sample = []
        current_households = []

        first_id = self.ids[0]
        previous_household = int(first_id[4])
        previous_minicell = int(first_id[2])
        previous_cell = int(first_id[0])

        household_counter = 0

        for s in self.ids:

            current_household = int(s[4])
            current_minicell = int(s[2])
            current_cell = int(s[0])

            if (current_minicell == previous_minicell) and (current_cell == previous_cell) and (current_household == previous_household):

                household_counter += 1

            else:

                check_household_cap(household_counter, self.household_cap, new_sample)

                current_households = []
                household_counter = 1
                
                previous_cell = current_cell
                previous_minicell = current_minicell
                previous_household = current_household

            current_households.append(s)

        new_sample = check_household_cap(household_counter, self.household_cap, new_sample)

        new_sample.sort()
        
        return new_sample
    


ids = ["0.0.0.0","0.0.0.1","0.0.0.2","0.0.0.3",
       "0.0.1.0","0.0.1.1",
       "0.0.2.0",
       "0.0.3.0","0.0.3.1","0.0.3.2","0.0.3.3","0.0.3.4",
       "0.1.0.0","0.1.0.1","0.1.0.2",
       "0.1.1.0","0.1.1.1",
       "2.0.0.0","2.0.0.1","2.0.0.2",
       "2.0.1.0","2.0.1.1",
       "2.1.0.0","2.1.0.1","2.1.0.2"]

random.shuffle(ids)

Test = SampleHouseholds(household_cap=2, ids=ids)

print(Test.cap_household())