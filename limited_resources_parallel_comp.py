import numpy as np
import epios
from scipy.interpolate import interp1d, make_interp_spline
import pandas as pd
import multiprocessing
import os
import matplotlib.pyplot as plt

def one_ite(dict):
    temp_store_path = dict['temp_store_path']
    job_id = dict['job_id']
    postprocess = dict['postprocess']
    k = dict['k']
    sample_size = dict['sample_size']
    time_sample = dict['time_sample']
    true_result_plot = dict['true_result_plot']
    demo_data = dict['demo_data']
    os.mkdir(temp_store_path + 'job_id_' + str(job_id))
    score_spline = []
    max_spline = []
    score_bspline = []
    max_bspline = []
    result, _ = postprocess.predict.AgeRegion(sample_size=sample_size,
                                                time_sample=time_sample,
                                                # comparison=True,
                                                #  non_responder=False,
                                                # gen_plot=True,
                                                sample_strategy='Random',
                                                data_store_path=temp_store_path + 'job_id_' + str(job_id) + '/')
    result_scaled = np.round(np.array(result[1]) * len(demo_data))
    cubic_interp = interp1d(time_sample, result_scaled, kind='cubic')
    y_interpolated = cubic_interp(range(max(time_sample) + 1))
    diff_interp = np.abs(np.array(true_result_plot) - y_interpolated)
    score_spline.append(np.sum(diff_interp))
    max_spline.append(max(diff_interp))
    spline = make_interp_spline(time_sample, result_scaled, k=k, bc_type='natural')  # Natural boundary conditions
    y_interpolated = spline(range(max(time_sample) + 1))
    diff_interp = np.abs(np.array(true_result_plot) - y_interpolated)
    score_bspline.append(np.sum(diff_interp))
    max_bspline.append(max(diff_interp))
    result, _ = postprocess.predict.Region(sample_size=sample_size,
                                                time_sample=time_sample,
                                                # comparison=True,
                                                #  non_responder=False,
                                                # gen_plot=True,
                                                sample_strategy='Random',
                                                data_store_path=temp_store_path + 'job_id_' + str(job_id) + '/')
    result_scaled = np.round(np.array(result[1]) * len(demo_data))
    cubic_interp = interp1d(time_sample, result_scaled, kind='cubic')
    y_interpolated = cubic_interp(range(max(time_sample) + 1))
    diff_interp = np.abs(np.array(true_result_plot) - y_interpolated)
    score_spline.append(np.sum(diff_interp))
    max_spline.append(max(diff_interp))
    spline = make_interp_spline(time_sample, result_scaled, k=k, bc_type='natural')  # Natural boundary conditions
    y_interpolated = spline(range(max(time_sample) + 1))
    diff_interp = np.abs(np.array(true_result_plot) - y_interpolated)
    score_bspline.append(np.sum(diff_interp))
    max_bspline.append(max(diff_interp))
    result, _ = postprocess.predict.Age(sample_size=sample_size,
                                                time_sample=time_sample,
                                                # comparison=True,
                                                #  non_responder=False,
                                                # gen_plot=True,
                                                sample_strategy='Random',
                                                data_store_path=temp_store_path + 'job_id_' + str(job_id) + '/')
    result_scaled = np.round(np.array(result[1]) * len(demo_data))
    cubic_interp = interp1d(time_sample, result_scaled, kind='cubic')
    y_interpolated = cubic_interp(range(max(time_sample) + 1))
    diff_interp = np.abs(np.array(true_result_plot) - y_interpolated)
    score_spline.append(np.sum(diff_interp))
    max_spline.append(max(diff_interp))
    spline = make_interp_spline(time_sample, result_scaled, k=k, bc_type='natural')  # Natural boundary conditions
    y_interpolated = spline(range(max(time_sample) + 1))
    diff_interp = np.abs(np.array(true_result_plot) - y_interpolated)
    score_bspline.append(np.sum(diff_interp))
    max_bspline.append(max(diff_interp))
    result, _ = postprocess.predict.Base(sample_size=sample_size,
                                                time_sample=time_sample,
                                                # comparison=True,
                                                #  non_responder=False,
                                                # gen_plot=True,
                                                sample_strategy='Random',
                                                data_store_path=temp_store_path + 'job_id_' + str(job_id) + '/')
    result_scaled = np.round(np.array(result[1]) * len(demo_data))
    cubic_interp = interp1d(time_sample, result_scaled, kind='cubic')
    y_interpolated = cubic_interp(range(max(time_sample) + 1))
    diff_interp = np.abs(np.array(true_result_plot) - y_interpolated)
    score_spline.append(np.sum(diff_interp))
    max_spline.append(max(diff_interp))
    spline = make_interp_spline(time_sample, result_scaled, k=k, bc_type='natural')  # Natural boundary conditions
    y_interpolated = spline(range(max(time_sample) + 1))
    diff_interp = np.abs(np.array(true_result_plot) - y_interpolated)
    score_bspline.append(np.sum(diff_interp))
    max_bspline.append(max(diff_interp))
    for file in ['pop_dist.json', 'microcells.csv', 'data.csv']:
        if os.path.exists(temp_store_path + 'job_id_' + str(job_id) + '/' + file):
            os.remove(temp_store_path + 'job_id_' + str(job_id) + '/' + file)
    os.rmdir(temp_store_path + 'job_id_' + str(job_id))
    return score_spline, max_spline, score_bspline, max_bspline

# Do prediction and comparison based age-region stratification

if __name__ == "__main__":  # Required for Windows

    score_spline_14days = []
    max_spline_14days = []
    score_bspline_14days = []
    max_bspline_14days = []
    score_spline_7days = []
    max_spline_7days = []
    score_bspline_7days = []
    max_bspline_7days = []
    num_ite = 100

    demo_data = pd.read_csv('./demographics_large.csv')
    time_data = pd.read_csv('./inf_status_history_large.csv')

    # Define the class instance
    postprocess = epios.PostProcess(time_data=time_data, demo_data=demo_data)
    time_sample=[0, 14, 28, 42, 56, 70, 84]
    sample_size = 750
    k = 3

    # This has the true result
    true_result_plot = []
    for t in range(max(time_sample) + 1):
        num = time_data.iloc[t, 1:].value_counts().get(3, 0)
        num += time_data.iloc[t, 1:].value_counts().get(4, 0)
        num += time_data.iloc[t, 1:].value_counts().get(5, 0)
        num += time_data.iloc[t, 1:].value_counts().get(6, 0)
        num += time_data.iloc[t, 1:].value_counts().get(7, 0)
        num += time_data.iloc[t, 1:].value_counts().get(8, 0)
        true_result_plot.append(num)

    temp_store_path = './temp'
    if not os.path.exists(temp_store_path):
        os.mkdir(temp_store_path)
    else:
        temp_store_path += '_'
        while os.path.exists(temp_store_path):
            temp_store_path += 'a'
        os.mkdir(temp_store_path)
    temp_store_path += '/'
    params_dict = {}
    params_dict['temp_store_path'] = temp_store_path
    params_dict['postprocess'] = postprocess
    params_dict['k'] = k
    params_dict['sample_size'] = sample_size
    params_dict['time_sample'] = time_sample
    params_dict['true_result_plot'] = true_result_plot
    params_dict['demo_data'] = demo_data
    iteration_inputs = []
    for i in range(num_ite):
        iteration_input = params_dict.copy()
        iteration_input['job_id'] = i
        iteration_inputs.append(iteration_input)

    # Create a pool of workers, the number of which is optional
    # By default, it will use the number of available CPU cores
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Map my_function across the data with the pool of workers
        results = pool.map(one_ite, iteration_inputs)
    for j in results:
        score_spline_14days += j[0]
        max_spline_14days += j[1]
        score_bspline_14days += j[2]
        max_bspline_14days += j[3]
    print('Mean score of Spline is', np.mean(score_spline_14days))
    print('Max of Spline is', max(max_spline_14days))
    print('Mean score of B-Spline is', np.mean(score_bspline_14days))
    print('Max of B-Spline is', max(max_bspline_14days))
    plt.figure()
    plt.hist(max_spline_14days, bins=20)
    plt.xlabel('Max Diff for spline (14 days)')
    plt.savefig('hist_max_diff_spline_14.png')
    plt.figure()
    plt.hist(max_bspline_14days, bins=20)
    plt.xlabel('Max Diff for B-spline (14 days)')
    plt.savefig('hist_max_diff_bspline_14.png')
    plt.figure()
    plt.hist(score_spline_14days, bins=20)
    plt.xlabel('Score for spline (14 days)')
    plt.savefig('hist_tv_spline_14.png')
    plt.figure()
    plt.hist(score_bspline_14days, bins=20)
    plt.xlabel('Score for B-spline (14 days)')
    plt.savefig('hist_tv_bspline_14.png')
    time_sample=[0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84]
    sample_size = 375
    params_dict = {}
    params_dict['temp_store_path'] = temp_store_path
    params_dict['postprocess'] = postprocess
    params_dict['k'] = k
    params_dict['sample_size'] = sample_size
    params_dict['time_sample'] = time_sample
    params_dict['true_result_plot'] = true_result_plot
    params_dict['demo_data'] = demo_data
    iteration_inputs = []
    for i in range(num_ite):
        iteration_input = params_dict.copy()
        iteration_input['job_id'] = i
        iteration_inputs.append(iteration_input)
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Map my_function across the data with the pool of workers
        results = pool.map(one_ite, iteration_inputs)
    for j in results:
        score_spline_7days += j[0]
        max_spline_7days += j[1]
        score_bspline_7days += j[2]
        max_bspline_7days += j[3]
    print('Mean score of Spline is', np.mean(score_spline_7days))
    print('Max of Spline is', max(max_spline_7days))
    print('Mean score of B-Spline is', np.mean(score_bspline_7days))
    print('Max of B-Spline is', max(max_bspline_7days))
    plt.figure()
    plt.hist(max_spline_7days, bins=20)
    plt.xlabel('Max Diff for spline (7 days)')
    plt.savefig('hist_max_diff_spline_7.png')
    plt.figure()
    plt.hist(max_bspline_7days, bins=20)
    plt.xlabel('Max Diff for B-spline (7 days)')
    plt.savefig('hist_max_diff_bspline_7.png')
    plt.figure()
    plt.hist(score_spline_7days, bins=20)
    plt.xlabel('Score for spline (7 days)')
    plt.savefig('hist_tv_spline_7.png')
    plt.figure()
    plt.hist(score_bspline_7days, bins=20)
    plt.xlabel('Score for B-spline (7 days)')
    plt.savefig('hist_tv_bspline_7.png')
    os.rmdir(temp_store_path)
