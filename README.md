# EpiOS
[![Operating systems](https://github.com/SABS-R3-Epidemiology/EpiOS/actions/workflows/os_versions.yml/badge.svg)](https://github.com/SABS-R3-Epidemiology/EpiOS/actions/workflows/os_versions.yml)
[![Python package](https://github.com/SABS-R3-Epidemiology/EpiOS/actions/workflows/python_versions.yml/badge.svg)](https://github.com/SABS-R3-Epidemiology/EpiOS/actions/workflows/python_versions.yml)
[![Style tests (flake8)](https://github.com/SABS-R3-Epidemiology/EpiOS/actions/workflows/style.yml/badge.svg)](https://github.com/SABS-R3-Epidemiology/EpiOS/actions/workflows/style.yml)
[![Documentation Status](https://readthedocs.org/projects/epios/badge/?version=latest)](https://epios.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/SABS-R3-Epidemiology/EpiOS/graph/badge.svg?token=FFZVJBPNM1)](https://codecov.io/gh/SABS-R3-Epidemiology/EpiOS)

## General Information
This project consists different methods to sample the population and evaluation of different methods. We include a lot of situations that may cause bias to the estimate of infection level based on the sample, including non-responders, false positive/negative rate, the ability of transmission profile for patients during their period of infection. Based on the [EpiABM](https://github.com/SABS-R3-Epidemiology/epiabm) model, this package can also output the best sampling method by running simulations of disease transmission to see the prediction error of each sampling method.

## Installation

EpiOS is not yet available on [PyPI](https://pypi.org/), but the module can be pip installed locally. The directory should first be downloaded to your local machine, and can then be installed using the command:

```console
pip install -e .
```

We also recommend you to install the [EpiABM](https://github.com/SABS-R3-Epidemiology/epiabm) model to generate the data of infection simulation. You can firstly download the [pyEpiabm](https://github.com/SABS-R3-Epidemiology/epiabm/tree/main/pyEpiabm) to any location on your machine, and can then be installed using the command:

```console
pip install -e path/to/pyEpiabm
```

## Documentation

 Documentations can be accessed via the above `docs` badge.

## Class Overview

 Here is a UML class diagram for our project:
 ![UML class diagram](./EpiOS.vpd.png)

## Configuration

 The `params.py` file include all parameters required in this model.

## Use `PostProcess` to generate plots

 Fistly, you need to define a new `PostProcess` object and input the demographical data `demodata` and infection data `timedata` generated from pyEpiabm.
 Secondly, you can used PostProcess.predict to perform prediction based on different sampling methods. You can directly call the sampling method you want to use as a method; then specify the time points to sample and sample size. Here, we will use `AgeRegion` as sampling method, `[0, 1, 2, 3, 4, 5]` as time points to be sampled and `3` to be the sample size.
 Lastly, you can specify whether you want to consider non-responders and whether youwant to compare your results with the true data by specifying the parameter `non_responder` and `comparison`.
 
 For code example, you can see the following:

 ```console
 python
 ```

 ```python
 import epios
 postprocess = epios.PostProcess(time_data=timedata, demo_data=demodata)
 
 res, diff = postprocess.predict.AgeRegion(
   time_sample=[0, 1, 2, 3, 4, 5], sample_size=3,
   non_responders=False,
   comparison=True,
   gen_plot=True,
   saving_path_sampling='path/to/save/sampled/predicted/infection/plot',
   saving_path_compare='path/to/save/comparison/plot'
   )
 ```

 Now, you will have your figure saved to the given path!

## Use `PostProcess` to select the best sampling method

 Fistly, you need to define a new `PostProcess` object and input the demographical data `demodata` and infection data `timedata` generated from pyEpiabm.
 Secondly, you can used PostProcess.best_method to compare performance of different sampling methods. You can provide methods you want to compare; then specify the sampling intervals to sample and sample size.
 Thirdly, you can specify whether you want to consider non-responders and whether youwant to compare your results with the true data by specifying the parameter `non_responder` and `comparison`.
 Additionally, since sampling methods are stochastic, you can specify the number of iterations run to gain the average performance. Moreover, `parallel_computation` can be turned on to speed up.
 Lastly, you can turn on the `hyperparameter_autotune` to automatically find the best combination of hyperparameters.
 
 For code example, you can see the following:

 ```console
 python
 ```

 ```python
 import epios
 postprocess = epios.PostProcess(time_data=timedata, demo_data=demodata)
 
 # Define the input keywards for finding the best method
 best_method_kwargs = {
     'age_group_width_range': [14, 17, 20]
 }
    
 # Suppose we want to compare among methods Age-Random, Base-Same,
 # Base-Random, Region-Random and AgeRegion-Random

 # And suppose we want to turn on the parallel computation to speed up
   if __name__ == '__main__':
      # This 'if' statement can be omitted when not using parallel computation
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
         sampling_interval=7,
         iteration=1,
         # When considering non-responders, input the following line
         # non_resp_rate=0.1,
         metric='mean',
         parallel_computation=True,
         **best_method_kwargs
      )
   # Then the output will be printed
