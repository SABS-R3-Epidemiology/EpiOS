import random
import numpy as np

#household_cap = ["cap_number", 3] 
household_cap = ["cap_proportion", 0.7]

# would loop through each cell and minicell
cell = 0
minicell = 0

ids = ["0.0.0.0","0.0.0.1","0.0.0.2",
       "0.0.1.0","0.0.1.1","0.0.1.2","0.0.1.3",
       "0.0.2.0","0.0.2.1","0.0.2.2",
       "0.0.3.0","0.0.3.1","0.0.3.2",
       "0.0.4.0","0.0.4.1","0.0.4.2"]
        
# list containing total no. people in each household
households = []

# storage variable
previous_household = 0

# store tally of people in each household
num_people = 0

# loop through every id
for i in ids:
    
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

# limit number of people sampled for each household
if household_cap[0] == "cap_number":

    # max number of people
    max_num = household_cap[1]

    # list of ids to sample from
    id_samples = []

    household_num = 0
    # loop over every household
    for h in households:

        
        # check if household has more people than max to sample from
        if h > max_num:

            # select random ids from the hosuehold to sample from
            sample = random.sample(range(0, h), max_num)

        else:
            
            sample = list(range(h))
            
        for s in range(len(sample)):

            id_samples.append(str(cell)+"."+str(minicell)+"."+str(household_num)+"."+str(sample[s]))


        household_num += 1
        
        
    print(id_samples)
    

# limit proportion of people sampled across households
elif household_cap[0] == "cap_proportion":
    
    sample_proportion = household_cap[1]


