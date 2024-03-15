import epios
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import math


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
                              false_positive=0.034,
                              false_negative=0.096,
                              stats_start_time=0,
                              method="Base",
                              plots="Analysis"):
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
        method (string): the sampler method used ("Base","Age","Region")
        plots (string): the plots produced ("Analysis","Ratios)

    Returns:
        None
    """

    # make sure start time is an appropiate number
    try:

        start_index = int(stats_start_time)

    except:

        print("Must use integer start time")
        return None
    
    if start_index >= len(sample_times):

        print("Must be a time within sample time range")
        return None

    # get starting and ending sample sizes
    start_sample_size = sample_range[0]
    end_sample_size = sample_range[1]

    # generate logarithmically spaced sample sizes using natural logarithm
    log_sample_sizes = np.logspace(np.log(start_sample_size), np.log(end_sample_size), num=num_samples,
                                   endpoint=True, base=np.e, dtype=int)

    #  iterate over the sample sizes
    for sample_size in log_sample_sizes:



        # initialise empty lists
        total_differences = np.zeros(len(sample_times))
        total_square_differences = total_differences.copy()
        total_sensitivity = np.zeros(len(sample_times))
        total_specificity = np.zeros(len(sample_times))
        total_test_accuracy = np.zeros(len(sample_times))
        total_sens_ratio = np.zeros(len(sample_times))
        total_spec_ratio = np.zeros(len(sample_times))
        total_percentage_estimated = np.zeros(len(sample_times))

        # initialise iteration count
        iteration_count = 0

        # run multiple iterations
        for _ in range(num_iterations):

            # display info to user on current iteration
            print(f"Sample Size {sample_size} : Performing Iteration {iteration_count + 1}")

            if method == "Base":

                # perform sampling in epios
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

            elif method == "Age":

                # perform sampling in epios
                result = postprocess.predict.Age(sample_size=sample_size,
                                                    time_sample=sample_times,
                                                    comparison=False,
                                                    gen_plot=False,
                                                    sample_strategy='Random',
                                                    saving_path_sampling=
                                                        './output/sample_plot',
                                                    saving_path_compare=
                                                        './output/compare_plot',
                                                    false_positive=false_positive,
                                                    false_negative=false_negative)

            elif method == "Region":

                # perform sampling in epios
                result = postprocess.predict.Region(sample_size=sample_size,
                                                    time_sample=sample_times,
                                                    comparison=False,
                                                    gen_plot=False,
                                                    sample_strategy='Random',
                                                    saving_path_sampling=
                                                        './output/sample_plot',
                                                    saving_path_compare=
                                                        './output/compare_plot',
                                                    false_positive=false_positive,
                                                    false_negative=false_negative)
            # filter result and remove sample times
            result = result[0]
            result = result[1:][0]

            # recover population size
            pop_size = len(demo_data)

            # calculate number of people in each category
            # TP: True Positive
            # FP: False Positive
            # TN: True Negative
            # FN: False Negative
            TP =[r[0] * pop_size for r in result]
            FP =[r[1] * pop_size for r in result]
            TN =[r[2] * pop_size for r in result]
            FN =[r[3] * pop_size for r in result]

            # calculate # people sampler believes are infected
            est_infected = [TP[i] + FP[i] for i in range(len(TP))]

            # calculate # people who are actually infected
            act_infected = [TP[i] + FN[i] for i in range(len(TP))]

            percentage_estimated = [0 if act_infected[i] == 0
                                    else 100 * est_infected[i] / act_infected[i] for i in range(len(act_infected))]

            # ratios to test behaviour of sensitivity, specificity
            sens_ratio, spec_ratio = get_ratios(TP, TN, FP, FN)

            # calculate PCR test stats
            sensitivity, specificity, test_accuracy = get_pcr_stats(TP, TN, FP, FN)

            # add to the ratio totals
            total_sens_ratio = [total_sens_ratio[i] + sens_ratio[i] for i in range(len(sens_ratio))]
            total_spec_ratio = [total_spec_ratio[i] + spec_ratio[i] for i in range(len(spec_ratio))]

            total_percentage_estimated = [total_percentage_estimated[i] + percentage_estimated[i] for i in range(len(total_percentage_estimated))]


            # add to the stat totals
            total_sensitivity = [total_sensitivity[i] + sensitivity[i] for i in range(len(sensitivity))]
            total_specificity = [total_specificity[i] + specificity[i] for i in range(len(specificity))]
            total_test_accuracy = [total_test_accuracy[i] + test_accuracy[i] for i in range(len(test_accuracy))]
   
            # calculate differences between estimate and actual, and add to total
            difference = [est_infected[i] - act_infected[i] for i in range(len(est_infected))]
            total_differences = [total_differences[i] + difference[i] for i in range(len(difference))]

            # calculate expected difference from probabilities
            #average_difference = np.ones(len(sample_times)) * (abs(false_positive - false_negative)) * pop_size

            # calculate square differences between estimate and actual, and add to total
            square_difference = [(est_infected[i] - act_infected[i])**2 for i in range(len(est_infected))]
            total_square_differences = [total_square_differences[i] + square_difference[i] for i in range(len(square_difference))]

            # increment iteration count
            iteration_count += 1

        # calculate averages of the above quantities by dividing by num_iterations
        av_difference = [total_differences[i] / num_iterations for i in range(len(total_differences))]
        av_square_difference = [total_square_differences[i] / num_iterations for i in range(len(total_square_differences))]
        absolute_difference = [abs(av_difference[i]) for i in range(len(av_difference))]
        av_sensitivity = [total_sensitivity[i] / num_iterations for i in range(len(total_sensitivity))]
        av_specificity = [total_specificity[i] / num_iterations for i in range(len(total_specificity))]
        av_test_accuracy = [total_test_accuracy[i] / num_iterations for i in range(len(total_test_accuracy))]
        av_sens_ratio = [total_sens_ratio[i] / num_iterations for i in range(len(total_sens_ratio))]
        av_spec_ratio = [total_spec_ratio[i] / num_iterations for i in range(len(total_spec_ratio))]
        av_percentage_estimated = [total_percentage_estimated[i] / num_iterations for i in range(len(total_percentage_estimated))]

        # calculate rmse; root mean squared error
        rmse = [np.sqrt(av_square_difference[i]) for i in range(len(av_square_difference))]

        if plots == "Analysis":

            # set-up plot pane figure
            fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(12, 8), tight_layout=True)

            # plot graphs on respective figures
            ax1.plot(sample_times, act_infected, label=f"Actual, Sample Size: {sample_size}", color='cyan')
            ax1.plot(sample_times, est_infected, label=f"Estimated, Sample Size: {sample_size}", color='olive')
            ax3.plot(sample_times, absolute_difference, label=f"Absolute Difference, Sample Size: {sample_size}", color='blue')
            ax3.plot(sample_times, rmse, label=f"RMSE, Sample Size: {sample_size}", color='red')
            #ax3.plot(sample_times, average_difference, label="Expected Absolute Difference")
            ax2.plot(sample_times[start_index :], av_sensitivity[start_index :], label=f"Sensitivity, Sample Size: {sample_size}")
            ax2.plot(sample_times[start_index :], av_specificity[start_index :], label=f"Specificity, Sample Size: {sample_size}")
            ax2.plot(sample_times[start_index :], av_test_accuracy[start_index :], label=f"Accuracy, Sample Size: {sample_size}")

            # add figure labels
            ax1.set_xlabel('Time (days)')
            ax2.set_xlabel('Time (days)')
            ax3.set_xlabel('Time (days)')
            ax1.set_ylabel('# Infected')
            ax2.set_ylabel('Percentage')
            ax3.set_ylabel('# Infected')
            ax1.set_title('# Infected: Estimated vs Actual')
            ax2.set_title('Test Statistics')
            ax3.set_title('#Infected: Estimated vs Actual')

            # text on figure showing parameters
            parameters_text = f"""Parameters: \n
            Sample Size: {sample_size} \n
            Number of Iterations: {num_iterations} \n
            Probability False Positive: {false_positive} \n
            Probability False Negatives: {false_negative} \n
            Sampler Method: {method}
            """

            # add text to figure
            ax4.text(0.2, 0.6, parameters_text, horizontalalignment='left',
            verticalalignment='center', transform=ax4.transAxes)

            # add plot legends
            ax1.legend()
            ax2.legend()
            ax3.legend()

            # save figures
            plt.savefig(f'{path}/truefalsepos_samplesize_{sample_size}.png')


        elif plots == "Ratios":

            fig2, ax21 = plt.subplots()

            ax21.plot(sample_times[start_index :], av_sens_ratio[start_index :], label=f"FN/TP, Sample Size: {sample_size}")
            ax21.plot(sample_times[start_index :], av_spec_ratio[start_index :], label=f"FP/TN, Sample Size: {sample_size}")
            ax21.set_xlabel('Time (days)')
            ax21.set_ylabel('Ratios')
            ax21.set_title('Ratios over time')
            ax21.legend()

            plt.savefig(f'{path}/truefalsepos_ratios_samplesize_{sample_size}.png')

        elif plots == "Percentages":

            fig, ax = plt.subplots()

            ax.plot(sample_times, av_percentage_estimated, label=f"Percentage estimated, Sample Size: {sample_size}")
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Percentage')
            ax.set_title('Percentage Estimated over time')
            ax.legend()

            plt.savefig(f'{path}/truefalsepos_percentages_samplesize_{sample_size}.png')

    return None


def get_ratios(TP, TN, FP, FN):

    sens_ratio = [np.inf if TP[i] == 0 else
                  FN[i] / TP[i] for i in range(len(TP))]
    
    spec_ratio = [np.inf if TN[i] == 0 else
                  FP[i] / TN[i]  for i in range(len(TP))]
    
    return sens_ratio, spec_ratio


def get_pcr_stats(TP, TN, FP, FN):

    # calculate PCR test stats
    sensitivity = [0 if TP[i] == 0 and FN[i] > 0 else 
                   1 if FN[i] == 0 else
                   TP[i] / (TP[i] + FN[i]) for i in range(len(TP))]

    specificity = [0 if TP[i] == 0 and FP[i] > 0 else
                   1 if FP[i] == 0 else
                   TN[i] / (TN[i] + FP[i]) for i in range(len(TP))]

    test_accuracy = [(TP[i] + TN[i])/(TP[i] + TN[i] + FP[i] + FN[i]) for i in range(len(TP))]

    return sensitivity, specificity, test_accuracy


def falseposneg_model(false_positive, false_negative, infectiousness):
    """Function to model the false positive/negative probabilities as a function
    of the infectiousness (viral load) of a person

    Args:
        false_positive (float): probability of a false positive
        false_negative (float): probability of a false negative
        infectiousness (float): viral load (normalised)

    Returns:
        float, float: updated probabilities
    """
    if math.isnan(infectiousness):

        infectiousness = 0

    #print(infectiousness)
    # selected model
    model_name = "None"

    # model 1
    if model_name == "model 1":

        # linear in infectiousness

        false_positive = (1 - infectiousness) * false_positive
        false_negative = (1 + infectiousness) * false_negative

    # model 2
    elif model_name == "model 2":


        # quadratic in infectiousness
        m = 2 * (infectiousness - 0.5)**2 + 0.5

        false_positive = (1 + m) * false_positive
        false_negative = (1 + m) * false_negative

    #model 3
    elif model_name == "model 3":


        # linear for people with viral load
        if infectiousness > 0:

            false_positive = 0.8 * false_positive
            false_negative = 1.2 * false_negative


    elif model_name == "model 4":

        if 0 < infectiousness <= 0.1:

            false_negative = 3 * false_negative

        elif infectiousness > 0.1:

            false_positive = 3 * false_positive


    elif model_name == "model 5":

        #' https://royalsocietypublishing.org/doi/10.1098/rsif.2020.0947'
        alpha1 = 0
        beta1 = 1
        false_positive = 0.9 / (1 + math.exp(-alpha1 -beta1 * infectiousness)) * false_positive

        alpha2 = -0.1
        beta2 = 2
        false_negative = np.exp(alpha2) * math.pow(infectiousness, beta2) * false_negative
        #print(false_positive)

    # return probabilities
    return false_positive, false_negative


def normalise_inf_data(path):

    if not os.path.exists(f'{path}/normalised_inf_data.csv'):
        # Load the original data from inf_data.csv
        inf_data = pd.read_csv(f'{path}/infectiousness_history.csv')

        # Create a MinMaxScaler object
        scaler = MinMaxScaler()

        # Normalize each column individually
        normalized_data = inf_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        # Save the normalized data to normalised_inf_data.csv
        normalized_data.to_csv(f'{path}/normalised_inf_data.csv', index=False)


# main script
if __name__ == "__main__":

    # get demographic and time data for Gibraltar
    path = './basic_infection_history_simulation_ex'
    demo_data = pd.read_csv(f'{path}/demographics.csv')
    time_data = pd.read_csv(f'{path}/inf_status_history.csv')
    #inf_data = pd.read_csv(f'{path}/infectiousness_history.csv')

    normalise_inf_data(path=path)

    inf_data = pd.read_csv(f'{path}/normalised_inf_data.csv')

    # define the PostProcess class instance
    postprocess = epios.PostProcess(time_data=time_data, demo_data=demo_data, inf_data=inf_data, model=falseposneg_model)

    # get sample times
    sample_times = [t for t in range(0, 61)]

    # bool to turn-on analysis
    run_analysis = True

    if run_analysis:

        # call analysis function
        test = analyse_imperfect_testing(sample_times=sample_times, 
                                                    sample_range=[5, 100], 
                                                    num_samples=3, 
                                                    num_iterations=100, 
                                                    false_positive=0.034,
                                                    false_negative=0.096,
                                                    stats_start_time=20,
                                                    method="Base",
                                                    plots="Analysis")
        