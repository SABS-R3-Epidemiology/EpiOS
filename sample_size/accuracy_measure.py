import epios
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_incidence_rates(infections, population_size):
    """Function to get the incidence rates of an infected population

    Args:
        infections (list): list of the number of infected people at each time-step
        population_size (int): size of the population

    Returns:
        list: list of the incidence rates at each time-step
    """

    new_cases = [infections[i+1] - infections[i] for i in range(0, len(infections) - 1)]
    new_cases.insert(0, true_result[0])
    new_cases = [n if n >= 0 else 0 for n in new_cases]

    population_at_risk = [population_size - infections[i] for i in range(0, len(infections))]

    incidence_rates = [new_cases[i] / population_at_risk[i] for i in range(0, len(new_cases))]

    return incidence_rates


def get_incidence_errors(diff, true_result, population_size):

    actual_incidence_rates = get_incidence_rates(true_result, population_size)

    result_scaled = true_result - diff

    predicted_incidence_rates = get_incidence_rates(result_scaled, population_size)

    difference_incidence_rates = [predicted_incidence_rates[i] - actual_incidence_rates[i] for i in range(0, len(predicted_incidence_rates))]

    return difference_incidence_rates





# This assumes the python venv is installed under epios folder
demo_data = pd.read_csv('./example/demographics.csv')
time_data = pd.read_csv('./example/inf_status_history.csv')

# Define the class instance
postprocess = epios.PostProcess(time_data=time_data, demo_data=demo_data)

population_size = len(demo_data) - 1

sample_times = [t for t in range(0, 30)]

# How does the number of people sampled affect the accuracy of the sample
# (measured at the percentage error on the incidence rate/total infection
# number)

sample_sizes = [10, 20, 30, 40, 50]
percentage_errors = [] # Maybe a matrix for time and sample size?

#  Iterate over the sample sizes
for sample_size in sample_sizes:
    # Do prediction and comparison
    result, diff, true_result = postprocess.predict.Base(sample_size=sample_size,
                                            time_sample=sample_times,
                                            comparison=True,
                                            gen_plot=False,
                                            sample_strategy='Random',
                                            saving_path_sampling=
                                                './output/sample_plot',
                                            saving_path_compare=
                                                './output/compare_plot',
                                            get_true_result=True)





    # percentage_errors = [100 * difference_incidence_rate[i] / (actual_incidence_rates[i]) 
    #                      for i in range(0, len(difference_incidence_rate))]

    errors = get_incidence_errors(diff, true_result, population_size)

    # Calculate the incidence rate where it is new cases divided by the
    # population at risk - minimum is 0
    # Require new cases for each time interval for both predicted and actual
    # Denominator must be population at risk - maybe Susceptible and Recovered?
    # For person-time, we must record how long they have been at risk during
    # that time-period. If only pointwise then this will not matter as time-
    # period is only 1 day

    # Save the percentage error
    #percentage_errors.append(percentage_error)

    plt.plot(sample_times, [abs(e) for e in errors])

# Create a plot of how the error changes as the sample size changes
# Usually would plot the incidence rate against time, but we want to show
# percentage error of the incidence rate between predicted and actual

# Plot a line for each sample size


plt.xlabel('Time')
plt.ylabel('Error')
plt.title('Error vs Sample Size')
plt.legend()
plt.show()

