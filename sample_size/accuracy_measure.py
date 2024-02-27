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
num_iterations = 5

#  Iterate over the sample sizes
for sample_size in sample_sizes:
    average = np.zeros(len(sample_times))
    # Run multiple iterations
    for _ in range(num_iterations):  # Change the number of iterations as needed
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

        iteration_errors = get_incidence_errors(diff, true_result, population_size)
        average = [average[i] + iteration_errors[i] for i in range(0, len(iteration_errors))]
        
    average = [average[i]/num_iterations for i in range(0, len(average))]
    plt.plot(sample_times, [abs(e) for e in average], label=f'Sample Size: {sample_size}')

plt.xlabel('Time')
plt.ylabel('Error')
plt.title('Error vs Sample Size')
plt.legend()
plt.show()

