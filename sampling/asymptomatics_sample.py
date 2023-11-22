import numpy as np
import random

# Read in false negatives and total population
total_false_negatives = 500
total_population = 2500

# We have the proposed number of samples
proposed_num_samples = 48

# We multiply this by a factor to increase it by a proportion
num_samples = int(proposed_num_samples * (1 + total_false_negatives / total_population))

# Return the updated number of samples
print(num_samples)




