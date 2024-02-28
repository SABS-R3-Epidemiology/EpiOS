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


def filter_ppes(ppes):

    # Convert to NumPy array
    ppes = np.array(ppes)
    outlier_factor = 1.5

    total_error = []
    for col_index in range(ppes.shape[1]):

        column = ppes[:, col_index]  # Get the column
        
        # Calculate quartiles and IQR
        q1 = np.percentile(column, 25)
        q3 = np.percentile(column, 75)
        iqr = q3 - q1
        
        # Define outlier bounds
        lower_bound = q1 - outlier_factor * iqr
        upper_bound = q3 + outlier_factor * iqr
        
        # Filter out outliers
        column_mask = (column >= lower_bound) & (column <= upper_bound)
        ppes = ppes[column_mask]

        column_total = np.sum(ppes[:, col_index])
        total_error.append(column_total)

    return total_error


# This assumes the python venv is installed under epios folder
demo_data = pd.read_csv('./example/demographics.csv')
time_data = pd.read_csv('./example/inf_status_history.csv')

# Define the class instance
postprocess = epios.PostProcess(time_data=time_data, demo_data=demo_data)

population_size = len(demo_data) - 1

sample_times = [t for t in range(0, 91)]

filter_outliers = False

# How does the number of people sampled affect the accuracy of the sample
# (measured at the percentage error on the incidence rate/total infection
# number)

start_sample_size = 10  # Starting sample size
end_sample_size = 500  # Ending sample size
num_samples = 5  # Number of samples

# Generate logarithmically spaced sample sizes using natural logarithm
log_sample_sizes = np.logspace(np.log(start_sample_size), np.log(end_sample_size), num=num_samples, endpoint=True, base=np.e, dtype=int)
num_iterations = 5

#  Iterate over the sample sizes
for sample_size in log_sample_sizes:

    #total_error = np.zeros(len(sample_times))

    ppes = []
    total_error = np.zeros(len(sample_times))

    # Run multiple iterations
    for _ in range(num_iterations):  # Change the number of iterations as needed

        print(f"Performing Iteration Sample Size {sample_size}")
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

        #iteration_errors = get_incidence_errors(diff, true_result, population_size)
        prevalence_percentage_error = [100 * abs(diff[i]) / true_result[i] for i in range(0, len(diff))]

        if filter_outliers:

            ppes.append(prevalence_percentage_error)

        #average = [average[i] + iteration_errors[i] for i in range(0, len(iteration_errors))]
        else:

            total_error = [total_error[i] + prevalence_percentage_error[i] for i in range(0, len(prevalence_percentage_error))]
        
    #average = [average[i]/num_iterations for i in range(0, len(average))]
    if filter_outliers:

        total_error = filter_ppes(ppes)

    average_error = [total_error[i] / num_iterations for i in range(0, len(total_error))]



    plt.plot(sample_times, [e for e in average_error], label=f'Sample Size: {sample_size}')

plt.xlabel('Time')
plt.ylabel('Percentage Error')
plt.title('Percentage Error vs Sample Size for the Prevalence')
plt.legend()
plt.show()

