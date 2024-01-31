import epios
import pandas as pd
# import numpy as np

# demo_data = pd.DataFrame({'ID': ['0.0.0.0', '0.0.0.1', '0.0.1.0',
#                                     '0.1.0.0', '0.2.0.0', '1.0.0.0'],
#                             'age': [1, 81, 45, 33, 20, 60]})
# time_data = pd.DataFrame({'time': [0, 1, 2, 3, 4, 5],
#                             '0.0.0.0': ['InfectASympt'] * 6,
#                             '0.0.0.1': [0, 0, 0, 'InfectASympt', 'InfectASympt',
#                                         'InfectASympt'],
#                             '0.0.1.0': [0, 0, 'InfectASympt', 'InfectASympt',
#                                         'InfectASympt', 'InfectASympt'],
#                             '0.1.0.0': [0, 0, 'InfectASympt', 'InfectASympt',
#                                         'InfectASympt', 'InfectASympt'],
#                             '0.2.0.0': [0, 'InfectASympt', 'InfectASympt',
#                                         'InfectASympt', 'InfectASympt',
#                                         'InfectASympt'],
#                             '1.0.0.0': [0, 0, 0, 0, 0, 'InfectASympt']})

# This assumes the python venv is installed under epios folder
demo_data = pd.read_csv('./example/demographics_processed.csv')
time_data = pd.read_csv('./example/inf_status_history.csv')

# Define the class instance
postprocess = epios.PostProcess(time_data=time_data, demo_data=demo_data)

# Do prediction and comparison based age-region stratification
result, diff = postprocess.predict.AgeRegion(sample_size=3,
                                             time_sample=[0, 1, 2, 3],
                                             comparison=True,
                                             non_responder=False,
                                             gen_plot=True,
                                             sample_strategy='Random')

# Define the input keywards for finding the best method
if __name__ == '__main__':
    kwargs = {
        # 'num_age_group': 17,
        # 'num_age_group_range': [17],
        'age_group_width_range': [5],
        # 'sampling_percentage_range': [0.1],
        'proportion_range': [0.01],
        'threshold_range': [1]
    }
    postprocess.best_method(
        methods=[
            # 'Base-Same',
            # 'Base-Random',
            # 'Region-Random',
            'AgeRegion-Random'
        ],
        sample_size=3,
        hyperparameter_autotune=False,
        non_responder=False,
        sampling_interval=1,
        iteration=10,
        non_resp_rate=0.1,
        metric='mean',
        parallel_computation=False,
        **kwargs
    )
