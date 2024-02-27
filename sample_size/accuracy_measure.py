import epios
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This assumes the python venv is installed under epios folder
demo_data = pd.read_csv('./example/demographics.csv')
time_data = pd.read_csv('./example/inf_status_history.csv')

# Define the class instance
postprocess = epios.PostProcess(time_data=time_data, demo_data=demo_data)

sample_times = [t for t in range(0, 30)]

# How does the number of people sampled affect the accuracy of the sample
# (measured at the percentage error on the incidence rate/total infection
# number)

sample_sizes = [10, 20, 30, 40, 50]
percentage_errors = [] # Maybe a matrix for time and sample size?

#  Iterate over the sample sizes
for sample_size in sample_sizes:
    # Do prediction and comparison
    result, diff = postprocess.predict.Base(sample_size=sample_size,
                                            time_sample=sample_times,
                                            comparison=True,
                                            gen_plot=False,
                                            sample_strategy='Random',
                                            saving_path_sampling=
                                                './output/sample_plot',
                                            saving_path_compare=
                                                './output/compare_plot')

    # Calculate the incidence rate where it is new cases divided by the
    # population at risk - minimum is 0
    # Require new cases for each time interval for both predicted and actual
    # Denominator must be population at risk - maybe Susceptible and Recovered?
    # For person-time, we must record how long they have been at risk during
    # that time-period. If only pointwise then this will not matter as time-
    # period is only 1 day

    # Save the percentage error
    percentage_errors.append(percentage_error)

# Create a plot of how the error changes as the sample size changes
# Usually would plot the incidence rate against time, but we want to show
# percentage error of the incidence rate between predicted and actual

# Plot a line for each sample size


plt.xlabel('Time')
plt.ylabel('Percentage Error')
plt.title('Error vs Sample Size')
plt.legend()
plt.show()

