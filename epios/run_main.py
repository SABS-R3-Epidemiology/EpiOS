import params
from sampler import Sampler
from sampler_age_region import SamplerAgeRegion

# Sampler Instance
#SAMPLE = Sampler(num_age_group=2, data_path="./input/test_data.csv")
SAMPLE = SamplerAgeRegion(num_age_group=2, data_path="./input/test_data.csv", geoinfo_path='./input/microcells.csv')

# Sample over age and region
choice = SAMPLE.sample(2000)

# Length of age/region sample
original_sample_length = len(choice)

# Shrink sample according to household limits
new_sample = []
for s in choice:

    if SAMPLE.person_allowed(new_sample, str(s), threshold=params.household_threshold):

        new_sample.append(str(s))

# Length of new sample
final_sample_length = len(new_sample)

# Compare these lengths
print("Sample after age/region: ", original_sample_length)
print("Sample after household: ", final_sample_length)
