************
Post-process
************

.. currentmodule:: epios

Overview:

- :class:`PostProcess`


.. autoclass:: PostProcess
    :special-members: __call__
    :members: best_method

Here is an example of using `PostProcess`
-----------------------------------------

.. code-block:: python

    import epios
    import pandas as pd

    # Define the simulation output data
    demo_data = pd.read_csv('demographics_processed.csv')
    time_data = pd.read_csv('inf_status_history.csv')
    
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
        'age_group_width_range': [14, 17, 20]
    }
    
    # Suppose we want to compare among methods Age-Random, Base-Same,
    # Base-Random, Region-Random and AgeRegion-Random

    # And suppose we want to turn on the parallel computation to speed up
    if __name__ == '__main__':  # This line can be omitted when not using parallel computation
        postprocess.best_method(
            methods=[
                'Age',
                'Base-Same',
                'Base-Random',
                'Region-Random',
                'AgeRegion-Random'
            ],
            sample_size=3,
            hyperparameter_autotune=True,
            non_responder=False,
            sampling_interval=1,
            iteration=1,
            # When considering non-responders, input the following line
            # non_resp_rate=0.1,
            metric='mean',
            parallel_computation=True,
            **best_method_kwargs
        )
    # Then the output will be printed