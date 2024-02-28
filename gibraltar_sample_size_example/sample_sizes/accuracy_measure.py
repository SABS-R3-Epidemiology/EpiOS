import epios
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This assumes the python venv is installed under epios folder
path = './gibraltar_sample_size_example'
demo_data = pd.read_csv(f'{path}/simulation_outputs/demographics.csv')
time_data = pd.read_csv(f'{path}/simulation_outputs/inf_status_history.csv')

def get_incidence_rates(infections, true_result, population_size):
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
    """Function to remove outliers from each column of the ppes array and
    return the average percentage errors for each column

    Args:
        ppes (array): the prevalence percentage error arrays for each
        iteration

    Returns:
        array: the average errors at each time point without outliers
    """

    # Convert to NumPy array
    ppes = np.array(ppes)
    outlier_factor = 1.5

    total_error = []
    average_error = []
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
        column = column[column_mask]

        column_total = np.sum(column)
        total_error.append(column_total)
        average_error.append(total_error[col_index]/len(column))

    return average_error


def get_rmse(sample_range, num_samples, num_iterations):
    """Function to calculate the rmse errors between the sampled and actual
    infected number

    Args:
        sample_range (array): the start and beginning of the sample range
        num_samples (int): number of samples
        num_iterations (int): number of iterations each sample is averaged over

    Returns:
        array: rmse values
    """

    start_sample_size = sample_range[0]  # Starting sample size
    end_sample_size = sample_range[1]  # Ending sample size

    # Generate logarithmically spaced sample sizes using natural logarithm
    log_sample_sizes = np.logspace(np.log(start_sample_size), np.log(end_sample_size), num=num_samples, endpoint=True, base=np.e, dtype=int)

    #  Iterate over the sample sizes
    for sample_size in log_sample_sizes:

        #rmse_errors = np.zeros(len(sample_times))
        square_diff_total = np.zeros(len(sample_times))

        # Run multiple iterations
        for _ in range(num_iterations):  # Change the number of iterations as needed

            print(f"Performing Iteration Sample Size {sample_size}")
            result, diff, true_result = postprocess.predict.Base(sample_size=sample_size,
                                                time_sample=sample_times,
                                                comparison=True,
                                                gen_plot=False,
                                                sample_strategy='Random',
                                                #data_store_path=f'{path}/input/',
                                                saving_path_sampling=
                                                    './output/sample_plot',
                                                saving_path_compare=
                                                    './output/compare_plot',
                                                get_true_result=True)


            square_diff = [diff[i]**2 for i in range(len(diff))]

            square_diff_total = [square_diff_total[i] + square_diff[i] for i in range(len(square_diff))]

            #mean_square_diff = square_diff.mean()
            #rmse = np.sqrt(mean_square_diff)
            
            #rmse = [np.sqrt(((predict[i] - actual[i]) ** 2).mean()) for i in range(len(actual))]

            #rmse_errors = [rmse_errors[i] + rmse[i] for i in range(len(rmse_errors))]

        #average = [average[i]/num_iterations for i in range(0, len(average))]
            
        #av_rmse_errors = [rmse_errors[i] / num_iterations for i in range(len(rmse_errors))]
            
        mean_square_diff = [square_diff_total[i] / num_iterations for i in range(len(square_diff_total))]

        rmse = [np.sqrt(mean_square_diff[i]) for i in range(len(mean_square_diff))]


        plt.plot(sample_times, [e for e in rmse], label=f'Sample Size: {sample_size}')

    plt.xlabel('Time')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Sample Size for the Prevalence')
    plt.legend()
    plt.show()

    return rmse


def get_prevalence_percentage_error(sample_times, 
                                    sample_range, 
                                    num_samples, 
                                    num_iterations, 
                                    filter_outliers=True, 
                                    plot_prevalence=True):
    """Function to return the average prevalence percentage errors, averaged
    over a number of iterations and ran for a range of sample sizes. The
    function has an option to plot this against time

    Args:
        sample_times (array): the time points at which the function will
        sample at (days)
        sample_range (array): array containing the minimum and maximum
        sample sizes to choose from
        num_samples (int): the number of samples chosen from a log-scale
        within the sample_range
        num_iterations (int): the number of iterations ran and averaged over
        for each sample size
        filter_outliers (bool, optional): option to filter outliers out of the
        prevalence percentage error. Defaults to True.
        plot_prevalence (bool, optional): option to plot the prevalence
        percentage error against time for each sample size. Defaults to True.

    Returns:
        array: the average prevalence percdntage errors at each time point
    """

    start_sample_size = sample_range[0]  # Starting sample size
    end_sample_size = sample_range[1]  # Ending sample size

    # Generate logarithmically spaced sample sizes using natural logarithm
    log_sample_sizes = np.logspace(np.log(start_sample_size), np.log(end_sample_size), num=num_samples, endpoint=True, base=np.e, dtype=int)

    #  Iterate over the sample sizes
    for sample_size in log_sample_sizes:

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
            average_error = filter_ppes(ppes)

        else:
            average_error = [total_error[i] / num_iterations for i in range(0, len(total_error))]

        if plot_prevalence:
            plt.plot(sample_times, [e for e in average_error], label=f'Sample Size: {sample_size}')

    if plot_prevalence:

        plt.xlabel('Time')
        plt.ylabel('Percentage Error')
        plt.title('Percentage Error in Prevalence against Time for different Sample Size')
        plt.legend()
        plt.show()

    return average_error

# Define the class instance
postprocess = epios.PostProcess(time_data=time_data, demo_data=demo_data)

#population_size = len(demo_data) - 1

sample_times = [t for t in range(0, 91)]

filter_outliers = True

sample_range = [10, 100]
num_samples = 3
num_iterations = 5


#get_rmse(sample_range, num_samples, num_iterations)

prevalence_error = get_prevalence_percentage_error(sample_times=sample_times, 
                                                   sample_range=sample_range, 
                                                   num_samples=num_samples, 
                                                   num_iterations=num_iterations, 
                                                   filter_outliers=filter_outliers, 
                                                   plot_prevalence=True)
