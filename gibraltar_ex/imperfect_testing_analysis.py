import epios
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_imperfect_testing_rates(sensitivity=0.7, specificity=0.95, test_accuracy=0.87):
    """Function to calculate false positive/negative probabilities using known values
    for sensitivity, specificity and test accuracy

    Args:
        sensitivity (float, optional): sensitivity of PCR. Defaults to 0.7.
        specificity (float, optional): specificity of PCR. Defaults to 0.95.
        accuracy (float, optional): accuracy of PCR. Defaults to 0.87.

    Returns:
        dict: container for false positive and negative probabilities
    """

    # matrix equation variables
    alpha = sensitivity / (1 - sensitivity)
    beta = specificity / (1 - specificity)
    gamma = test_accuracy

    # set-up matrices
    M = np.array([[alpha, beta], [1, 1]])
    N = np.array([[gamma], [1 - gamma]])
    M_inverse = np.linalg.inv(M)

    # peform matrix multiplication
    result = np.matmul(M_inverse, N)

    # return proababilities
    false_negative = result[0][0]
    false_positive = result[1][0]

    return {"Probability False Positive": false_positive,
            "Probability False Negative": false_negative}


def analyse_imperfect_testing(sample_times, 
                              sample_range, 
                              num_samples, 
                              num_iterations, 
                              false_positive=0,
                              false_negative=0,
                              stats_start_time=0):
    """Function to produce graphs analysing the impact of imperfect testing

    Args:
        sample_times (array): the time points at which the function will
        sample at (days)
        sample_range (array): array containing the minimum and maximum
        sample sizes to choose from
        num_samples (int): the number of samples chosen from a log-scale
        within the sample_range
        num_iterations (int): the number of iterations ran and averaged over
        for each sample size
        false_positive (float): probability of false positives
        false_negative (float): probability of false negatives
        stats_start_time (int): the time from which the statistics are plotted

    Returns:
        None
    """

    try:

        start_index = int(stats_start_time)

    except:

        print("Must use integer start time")
        return None
    
    if start_index >= len(sample_times):

        print("Must be a time within sample time range")
        return None

    start_sample_size = sample_range[0]  # Starting sample size
    end_sample_size = sample_range[1]  # Ending sample size

    # Generate logarithmically spaced sample sizes using natural logarithm
    log_sample_sizes = np.logspace(np.log(start_sample_size), np.log(end_sample_size), num=num_samples,
                                   endpoint=True, base=np.e, dtype=int)

    #  Iterate over the sample sizes
    for sample_size in log_sample_sizes:


        fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)

        total_differences = np.zeros(len(sample_times))
        total_square_differences = total_differences.copy()


        total_sensitivity = np.zeros(len(sample_times))
        total_specificity = np.zeros(len(sample_times))
        total_test_accuracy = np.zeros(len(sample_times))

        # Run multiple iterations
        for _ in range(num_iterations):  # Change the number of iterations as needed

            print(f"Performing Iteration Sample Size {sample_size}")
            result = postprocess.predict.Base(sample_size=sample_size,
                                                time_sample=sample_times,
                                                comparison=False,
                                                gen_plot=False,
                                                sample_strategy='Random',
                                                saving_path_sampling=
                                                    './output/sample_plot',
                                                saving_path_compare=
                                                    './output/compare_plot',
                                                get_true_result=False,
                                                false_positive=false_positive,
                                                false_negative=false_negative)

            
            result = result[0]
            result = result[1:][0] # remove sample times from list

            pop_size = len(demo_data) # get population size

            TP =[r[0] * pop_size for r in result]
            FP =[r[1] * pop_size for r in result]
            TN =[r[2] * pop_size for r in result]
            FN =[r[3] * pop_size for r in result]

            est_infected = [TP[i] + FP[i] for i in range(len(TP))]
            act_infected = [TP[i] + FN[i] for i in range(len(TP))]

            
        

            sensitivity = [0 if TP[i] == 0 else TP[i] / (TP[i] + FN[i]) for i in range(len(TP))]
            specificity = [0 if TP[i] == 0 else TN[i] / (TN[i] + FP[i]) for i in range(len(TP))]
            test_accuracy = [(TP[i] + TN[i])/(TP[i] + TN[i] + FP[i] + FN[i]) for i in range(len(TP))]

            total_sensitivity = [total_sensitivity[i] + sensitivity[i] for i in range(len(sensitivity))]
            total_specificity = [total_specificity[i] + specificity[i] for i in range(len(specificity))]
            total_test_accuracy = [total_test_accuracy[i] + test_accuracy[i] for i in range(len(test_accuracy))]


            #est_infected = [r[0] * pop_size for r in result]      
            #act_infected = [r[1] * pop_size for r in result]    

            difference = [abs(est_infected[i] - act_infected[i]) for i in range(len(est_infected))]

            total_differences = [total_differences[i] + difference[i] for i in range(len(difference))]

            #average_difference = np.ones(len(sample_times)) * (abs(false_positive - false_negative)) * pop_size

            square_difference = [(est_infected[i] - act_infected[i])**2 for i in range(len(est_infected))]

            total_square_differences = [total_square_differences[i] + square_difference[i] for i in range(len(square_difference))]

        av_difference = [total_differences[i] / num_iterations for i in range(len(total_differences))]

        av_square_difference = [total_square_differences[i] / num_iterations for i in range(len(total_square_differences))]

        absolute_difference = [abs(av_difference[i]) for i in range(len(av_difference))]

        rmse = [np.sqrt(av_square_difference[i]) for i in range(len(av_square_difference))]

        av_sensitivity = [total_sensitivity[i] / num_iterations for i in range(len(total_sensitivity))]
        av_specificity = [total_specificity[i] / num_iterations for i in range(len(total_specificity))]
        av_test_accuracy = [total_test_accuracy[i] / num_iterations for i in range(len(total_test_accuracy))]

        ax1.plot(sample_times, act_infected, label=f"Actual, Sample Size: {sample_size}", color='cyan')
        ax1.plot(sample_times, est_infected, label=f"Estimated, Sample Size: {sample_size}", color='olive')
        ax3.plot(sample_times, absolute_difference, label=f"Absolute Difference, Sample Size: {sample_size}", color='blue')
        ax3.plot(sample_times, rmse, label=f"RMSE, Sample Size: {sample_size}", color='red')
        #plt.plot(sample_times, average_difference, label="Expected Absolute Difference")
        ax2.plot(sample_times[start_index :], av_sensitivity[start_index :], label=f"Sensitivity, Sample Size: {sample_size}")
        ax2.plot(sample_times[start_index :], av_specificity[start_index :], label=f"Specificity, Sample Size: {sample_size}")
        ax2.plot(sample_times[start_index :], av_test_accuracy[start_index :], label=f"Accuracy, Sample Size: {sample_size}")


        ax1.set_xlabel('Time (days)')
        ax2.set_xlabel('Time (days)')
        ax3.set_xlabel('Time (days)')

        ax1.set_ylabel('# Infected')
        ax2.set_ylabel('Percentage')
        ax3.set_ylabel('# Infected')

        ax1.set_title('# Infected: Estimated vs Actual')
        ax2.set_title('Test Statistics')
        ax3.set_title('#Infected: Estimated vs Actual')

        parameters_text = f"""Parameters: \n
        Sample Size: {sample_size} \n
        Number of Iterations: {num_iterations} \n
        Probability False Positive: {false_positive} \n
        Probability False Negatives: {false_negative}
        """

        ax4.text(0.2, 0.6, parameters_text, horizontalalignment='left',
        verticalalignment='center', transform=ax4.transAxes)


        ax1.legend()
        ax2.legend()
        ax3.legend()

        plt.savefig(f'{path}/truefalsepos_samplesize_{sample_size}.png')

    return None

# This assumes the python venv is installed under epios folder
path = './gibraltar_ex'
demo_data = pd.read_csv(f'{path}/demographics.csv')
time_data = pd.read_csv(f'{path}/inf_status_history.csv')

# Define the class instance
postprocess = epios.PostProcess(time_data=time_data, demo_data=demo_data)

sample_times = [t for t in range(0, 91)]

r = calculate_imperfect_testing_rates()

print(r)

run_testing = False

if run_testing:

    test = analyse_imperfect_testing(sample_times=sample_times, 
                                                sample_range=[100, 500], 
                                                num_samples=2, 
                                                num_iterations=4, 
                                                false_positive=0.034,
                                                false_negative=0.096,
                                                stats_start_time=20)


