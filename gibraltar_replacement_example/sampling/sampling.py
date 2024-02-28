import epios

from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = './gibraltar_replacement_example'
# This assumes the python venv is installed under epios folder
demo_data = pd.read_csv(f'{path}/simulation_outputs/demographics.csv')
time_data = pd.read_csv(f'{path}/simulation_outputs/inf_status_history.csv')


def predict_base(post_process: epios.PostProcess, sample_size: int,
                 time_sample: List[int], comparison: bool, gen_plot: bool,
                 sample_strategy: str, plot_dir: str) -> Tuple[np.array, ...]:
    """Uses the `Base` method from `epios.PostProcess` to plot and predict
    incidence rate from a sample.

    Returns
    -------
    Tuple[np.array, ...]
         Returns the times, true and predicted incidence rates in a tuple to be
         used for further analysis
    """

    # Plot paths
    sample_path = f'{plot_dir}/sample.png'
    compare_path = f'{plot_dir}/compare.png'

    # Make prediction
    result, diff = post_process.predict.Base(sample_size=sample_size,
                                             time_sample=time_sample,
                                             comparison=comparison,
                                             gen_plot=gen_plot,
                                             sample_strategy=sample_strategy,
                                             saving_path_sampling=sample_path,
                                             saving_path_compare=compare_path)

    # Find population size and get true data
    pop_size = len(post_process.demo_data)
    times, prediction = result[0], np.array(result[1]) * pop_size
    true = diff + prediction

    # This clears the previous plots
    plt.clf()
    return times, true, prediction


def plot(times_array: np.array, true_array: np.array,
         predicted_array: np.array, plot_dir: str):
    """Mimics plots produced in EpiOS - to separate the processes of generating
    data and plotting.

    Parameters
    ----------
    times_array : np.array
        Time steps
    true_array : np.array
        True incidence rate
    predicted_array : np.array
        Predicted result from sample
    plot_dir : str
        Directory for plots
    """
    plt.plot(times_array, predicted_array)
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title("Predicted infection numbers from the sample")
    plt.savefig(f"{plot_dir}/sample.png")
    plt.clf()
    plt.plot(times_array, predicted_array, true_array,
             np.abs(predicted_array - true_array),
             label=["Predicted result", "True result", "Absolute difference"])
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title("Number of infected")
    plt.savefig(f"{plot_dir}/compare.png")
    plt.clf()


def rand_and_same(post_process: epios.PostProcess, sample_size: int,
                  time_sample: List[int], comparison: bool,
                  path: str) -> Tuple[np.array, ...]:
    """This returns predictions from the Random and Same sampling strategies,
    without plotting. These will be averaged over multiple runs

    Returns
    -------
    Tuple[np.array, ...]
        Times array, true results, predictions with Random strategy and
        predictions with Same strategy
    """
    # Do prediction and comparison with Random sampling strategy - this samples
    # different people every time step
    base_random = predict_base(post_process=post_process,
                               sample_size=sample_size,
                               time_sample=time_sample,
                               comparison=comparison,
                               gen_plot=False,
                               sample_strategy='Random',
                               plot_dir=f'{path}/sampling/random_strategy')
    times, true, pred_random = base_random

    # Do prediction and comparison with Same sampling strategy - this tracks
    # the same people every time step
    pred_same = predict_base(post_process=post_process,
                             sample_size=sample_size,
                             time_sample=time_sample,
                             comparison=comparison,
                             gen_plot=False,
                             sample_strategy='Same',
                             plot_dir=f'{path}/sampling/same_strategy')[2]

    return times, true, pred_random, pred_same


# Initialise a post process
post_process = epios.PostProcess(demo_data=demo_data, time_data=time_data)

# Run the Random and Same strategies for num_iters iterations so that we can
# average over the results
random_predictions = []
same_predictions = []
num_iters = 50
for j in range(num_iters):
    predictions = rand_and_same(post_process=post_process,
                                sample_size=100,
                                time_sample=list(range(100)),
                                comparison=True,
                                path=path)
    times, true, pred_random, pred_same = predictions
    random_predictions.append(pred_random)
    same_predictions.append(pred_same)

    rmse_random = np.sqrt(((true - pred_random) ** 2).mean())
    rmse_same = np.sqrt(((true - pred_same) ** 2).mean())
    print(f"Root mean squared error for random sampling: {rmse_random}")
    print(f"Root mean squared error for same sampling: {rmse_same}")

random_array = np.array(random_predictions)
same_array = np.array(same_predictions)
mean_random_array = np.mean(random_array, axis=0)
same_random_array = np.mean(same_array, axis=0)

# Plot the means

