************
Post-process
************

.. currentmodule:: epios

Overview:

- :class:`PostProcess`


.. autoclass:: PostProcess
    :special-members: __call__
    :members: best_method

Here is an example of using `PostProcess`:

.. code-block:: python

    import epios
    import pandas as pd

    # Define the simulation output data
    demo_data = pd.DataFrame({'ID': ['0.0.0.0', '0.0.0.1', '0.0.1.0',
                                     '0.1.0.0', '0.2.0.0', '1.0.0.0'],
                              'age': [1, 81, 45, 33, 20, 60]})
    time_data = pd.DataFrame({'time': [0, 1, 2, 3, 4, 5],
                              '0.0.0.0': ['InfectASympt'] * 6,
                              '0.0.0.1': [0, 0, 0, 'InfectASympt', 'InfectASympt',
                                          'InfectASympt'],
                              '0.0.1.0': [0, 0, 'InfectASympt', 'InfectASympt',
                                          'InfectASympt', 'InfectASympt'],
                              '0.1.0.0': [0, 0, 'InfectASympt', 'InfectASympt',
                                          'InfectASympt', 'InfectASympt'],
                              '0.2.0.0': [0, 'InfectASympt', 'InfectASympt',
                                          'InfectASympt', 'InfectASympt',
                                          'InfectASympt'],
                              '1.0.0.0': [0, 0, 0, 0, 0, 'InfectASympt']})
    
    # Define the class instance
    postprocess = epios.PostProcess(time_data=time_data, demo_data=demo_data,
                                    data_store_path='./input/')
    
    # Define the input keywords for doing one single comparison
    sampling_kwargs = {
        'gen_plot': True,
        'sample_strategy': 'Random'
    }
    result, diff = postprocess(sampling_method='AgeRegion', sample_size=3,
                               time_sample=[0, 1, 2, 3], comparison=True,
                               non_responder=False, **sampling_kwargs)
    
    # Define the input keywards for finding the best method
    best_method_kwargs = {
        'age_group_width': [14, 17, 20]
    }
    
    if __name__ == '__main__':
        postprocess.best_method(
            methods=[
                'Base-Same',
                'Age-Same',
                'Region-Same',
                'AgeRegion-Same',
                'Base-Random',
                'Age-Random',
                'Region-Random',
                'AgeRegion-Random'
            ],
            sample_size=3,
            hyperparameter_autotune=True,
            non_responder=False,
            sampling_interval=1,
            metric='mean',
            iteration=100,
            **best_method_kwargs
        )