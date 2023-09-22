'''
This is the benchmarking module for discrete tabular data. This module supports methods that evaluates causal discovery algorithms against various challenges, 
such as their sample complexity, variable complexity, etc. Users can use either synthetically generated data, or provide their own data for benchmarking. 

The default evaluation metrics supported by this module are Precision, Recall, F1 Score, and Time Taken by the algorithm. There is also an option for users to 
include their own custom metrics when calling the benchmarking module.

We provide support for a default set of causal discovery algorithms. Users also have the option to include their own algorithm when calling the 
benchmarking module.

Data:

1. Synthetic data: this module randomly generates both the causal graph (and the corresponding structural equation model) and the data associated 
with it. This module supports several benchmarking methods which evaluate causasl discovery algorithms on various aspects such as sample complexity, 
variable complexity, graph sparsity, etc. Depending on what is being evaluated, the corresponding method generates the graphs and data accordingly. 
Synthetic data evaluation serves two purposes:

  a. compare the performance of each causal discovery algorithm across different values of a variant (e.g. increasing number of sample),

  b. compare the performance of different causal discovery algorithms for any on given value of a variant.

2. User provided data: In this case, since the data is fixed, this module helps evaluate the performance of one or more causal discovery algorithms 
on the provided data. Since the data is not synthetically generated, in order to compute the evaluation metrics such as Precision/Recall, we need the 
ground truth causal graph. Therefore, the user provided data accepted by this module must contain this information. Specifically, the data must be a list 
of tuples, where each tuple contains the triplet (data_array, var_names, graph_gt), where data_array is a 2D Numpy data array of shape (samples x variables), 
var_names is a list of variable names, and graph_gt is the ground truth causal graph in the form of a Python dictionary, where keys are the variable names, 
and the corresponding values are a list of parent names.
'''
import numpy as np
import matplotlib
from typing import Callable
from matplotlib import pyplot as plt
import pickle as pkl
from functools import partial
import time
from typing import TypedDict, Tuple, List, Union, Optional, Dict
import tqdm
from .base import base_synthetic_tabular_benchmark, BenchmarkDiscreteTabularBase

class BenchmarkDiscreteTabular(BenchmarkDiscreteTabularBase):
    '''
    Discrete tabular data benchmarking module. This class inherits the methods and 
    variables from BenchmarkTabularBase and BenchmarkDiscreteTabularBase, and defines 
    benchmarking methods that evaluates causal discovery algorithms against various challenges, 
    such as their sample complexity, variable complexity, etc.
    '''
    def __init__(self, algo_dict:Dict=None, kargs_dict:Dict=None, num_exp:int=20, custom_metric_dict:Optional[Dict]={}, **kargs):
        '''
        Discrete tabular data benchmarking module
        
        :param algo_dict: A Python dictionary where keys are names of causal discovery algorithms, and 
            values are the unistantiated class objects for the corresponding algorithm. Note that this class 
            must be inherited from the `BaseTabularAlgoFull` class that can be found in causalai.models.tabular.base.
            Crucially, this class constructor must take a `TabularData` object (found in causalai.data.tabular) as input, 
            and should have a `run` method which performs the causal discovery and returns a Python dictionary. The keys of this 
            dictionary should be of the form:

            {
                var_name1: {'parents': [par(var_name1)]},
                var_name2: {'parents': [par(var_name2)]}
            }

            where par(.) denotes the parent variable name of the argument variable name.
        :type algo_dict: Dict
        :param kargs_dict: A Python dictionary where keys are names of causal discovery algorithms (same as algo_dict), 
            and the corresponding values contain any arguments to be passed to the `run` method of the class object specified in 
            algo_dict.
        :type kargs_dict: Dict
        :param num_exp: The number of independent runs to perform per experiment, each with a different random seed. A different 
            random seed generates a different synthetic graph and data for any given configuration. Note that for use provided data, 
            num_exp is not used.
        :type num_exp: int
        :param custom_metric_dict: A Python dictionary for specifying custom metrics in addition to the default evaluation metrics 
            calculated for each experiment (precision, recall, F1 score, and time taken). The keys of this dictionary are the names 
            of the user specified metrics, and the corresponding values are callable functions that take as input (graph_est, graph_gt). 
            Here graph_est and graph_gt are the estimated and ground truth causal graph. These graphs are specified as Python Dictionaries, 
            where keys are the children names, and the corresponding values are lists of parent variable names.
        :type custom_metric_dict: Dict
        '''
        BenchmarkDiscreteTabularBase.__init__(self, algo_dict=algo_dict, num_exp=num_exp,
                                         kargs_dict=kargs_dict, custom_metric_dict=custom_metric_dict, **kargs)

    def benchmark_variable_complexity(self, num_vars_list:List[int] = [2,10,20,40], graph_density: float=0.1, T:int=1000,\
                           fn:Callable = lambda x:x, coef:float=0.1, noise_fn:Callable=np.random.randn):
        '''
        Variable Complexity: Benchmark algorithms on synthetic data with different number of variables. 
        The synthetic data for any variable is generated using a structural equation model (SEM) of the form:

        child = sum_i coef* parent_i + noise

        and then discretized by binning.
        
        :param num_vars_list: It contains list of number of variables to be used to generate synthetic data.
        :type num_vars_list: List[int]
        :param graph_density: Float value in (0,1] specifying the density of the causal graph. The value is used 
            as the probability with which an edge between 2 nodes exists during the causal graph generation process.
        :type graph_density: float
        :param T: Integer value specifying the number of samples in the generated data.
        :type T: int
        :param fn: Callable function that acts as the non-linearity on parent variable value in the structural 
            equation model. This same function is applied on all parents.
        :type fn: Callable
        :param coef: Coefficient for the parent variable value in the structural 
            equation model. This same coefficient is used for all the parents.
        :type coef: float
        :param noise_fn: Callable function from which noise is sampled in the structural 
            equation model. This same function is used in all the equations.
        :type noise_fn: Callable
        '''
        all_results = []

        self.variant_values = num_vars_list
        self.variant_name = 'Number of Variables'
        for num_vars in num_vars_list:
            noise_fn_list = [noise_fn]*num_vars
            result_list = base_synthetic_tabular_benchmark(self.algo_dict, 
                                              self.kargs_dict, noise_fn_list, num_vars=num_vars,
                                              graph_density=graph_density, T=T, num_exp=self.num_exp,
                                              fn = fn,
                                              coef=coef,
                                              discrete=True, nstates=5,
                                              custom_metric_dict=self.custom_metric_dict)
            all_results.append(result_list)
        self.results_full = all_results

    def benchmark_sample_complexity(self, T_list:List[int] = [100, 500,1000,5000], num_vars:int=20, graph_density:float=0.1,\
                           fn:Callable = lambda x:x, coef:float=0.1, noise_fn:Callable=np.random.randn):
        '''
        Sample Complexity: Benchmark algorithms on synthetic data with different number of samples. 
        The synthetic data for any variable is generated using a structural equation model (SEM) of the form:

        child = sum_i coef* parent_i + noise

        and then discretized by binning.
        
        :param T_list: It contains list of number of samples to be used to generate synthetic data.
        :type T_list: List[int]
        :param num_vars: Integer value specifying the number of variables in the generated data.
        :type num_vars: int
        :param graph_density: Float value in (0,1] specifying the density of the causal graph. The value is used 
            as the probability with which an edge between 2 nodes exists during the causal graph generation process.
        :type graph_density: float
        :param fn: Callable function that acts as the non-linearity on parent variable value in the structural 
            equation model. This same function is applied on all parents.
        :type fn: Callable
        :param coef: Coefficient for the parent variable value in the structural 
            equation model. This same coefficient is used for all the parents.
        :type coef: float
        :param noise_fn: Callable function from which noise is sampled in the structural 
            equation model. This same function is used in all the equations.
        :type noise_fn: Callable
        '''
        all_results = []

        self.variant_values = T_list
        self.variant_name = 'Number of Samples'
        for T in T_list:
            noise_fn_list = [noise_fn]*num_vars
            result_list = base_synthetic_tabular_benchmark(self.algo_dict, 
                                              self.kargs_dict, noise_fn_list, num_vars=num_vars,
                                              graph_density=graph_density, T=T, num_exp=self.num_exp,
                                              fn = fn,
                                              coef=coef,
                                              discrete=True,
                                              custom_metric_dict=self.custom_metric_dict)
            all_results.append(result_list)
        self.results_full = all_results

    def benchmark_graph_density(self, graph_density_list:List[float] = [0.05, 0.1, 0.2, 0.5], num_vars:int=20, T:int=1000,\
                           fn:Callable = lambda x:x, coef:float=0.1, noise_fn:Callable=np.random.randn):
        '''
        Graph density: Benchmark algorithms on synthetic data with different number of samples. 
        The synthetic data for any variable is generated using a structural equation model (SEM) of the form:

        child = sum_i coef* parent_i + noise

        and then discretized by binning.
        
        :param graph_density_list: It contains list of graph denity values to be used to generate the causal graph. 
            Each value must be in (0,1].
        :type graph_density_list: List[float]
        :param num_vars: Integer value specifying the number of variables in the generated data.
        :type num_vars: int
        :param T: Integer value specifying the number of samples in the generated data.
        :type T: int
        :param fn: Callable function that acts as the non-linearity on parent variable value in the structural 
            equation model. This same function is applied on all parents.
        :type fn: Callable
        :param coef: Coefficient for the parent variable value in the structural 
            equation model. This same coefficient is used for all the parents.
        :type coef: float
        :param noise_fn: Callable function from which noise is sampled in the structural 
            equation model. This same function is used in all the equations.
        :type noise_fn: Callable
        '''
        all_results = []

        self.variant_values = graph_density_list
        self.variant_name = 'Graph Density'
        for graph_density in graph_density_list:
            noise_fn_list = [noise_fn]*num_vars
            result_list = base_synthetic_tabular_benchmark(self.algo_dict, 
                                              self.kargs_dict, noise_fn_list, num_vars=num_vars,
                                              graph_density=graph_density, T=T, num_exp=self.num_exp,
                                              fn = fn,
                                              coef=coef,
                                              discrete=True, nstates=5,
                                              custom_metric_dict=self.custom_metric_dict)
            all_results.append(result_list)
        self.results_full = all_results


