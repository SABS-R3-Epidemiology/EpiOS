# number of samples
no_sample = 100
# false-positive rate
false_pos_rate = 1
# cap on people per household
household_threshold = 3
# Need to specify one of the following when defining a Sampler class
data = None
data_path = None
# The number of age groups
# Each age group have width 5 years old
num_age_group = 17
# Whether to turn on the non-responder function
# If non-responder function is turned on, need to specify non-responder ID
# for each sampling
nonResp = False
nonRespID = []
# The proportion of additional samples
# taken from a specific age-regional group caused by non-responders
sampling_percentage = 0.1
# The proportion of total groups to be sampled additionally
# caused by non-responders
proportion = 0.01
# The lowest number of age-regional groups to be sampled additionally
# caused by non-responders
threshold = None
