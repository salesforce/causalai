
import numpy as np
import matplotlib
from typing import Callable
from typing import TypedDict, Tuple, List, Union, Optional, Dict
from matplotlib import pyplot as plt
import pickle as pkl
from functools import partial
import time
import tqdm

from causalai.data.data_generator import DataGenerator, GenerateSparseTabularSEM

from causalai.models.tabular.pc import PC
from causalai.models.common.CI_tests.partial_correlation import PartialCorrelation
from causalai.models.common.CI_tests.discrete_ci_tests import DiscreteCI_tests

from causalai.models.tabular.ges import GES
from causalai.models.tabular.lingam import LINGAM

from causalai.data.tabular import TabularData
from causalai.data.transforms.time_series import StandardizeTransform

from causalai.misc.misc import get_precision_recall


### Base class and functions ###

class BenchmarkTabularBase:
    '''
    Base class for the tabular data benchmarking module for both continuous and discrete cases. This class defines 
    methods for aggregating and plotting results, and a method for benchmarking on a user provided list of datasets.
    '''
    def __init__(self, algo_dict:Dict=None, kargs_dict:Dict=None, num_exp:int=20, custom_metric_dict:Dict={}, **kargs):
        '''
        Base tabular data benchmarking module
        
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
        assert algo_dict is not None, f'algo_dict cannot be None. It must be a dictionary with algorithm name in keys and the algorithm class '\
                                      f'object as values.'
        self.algo_names = list(algo_dict.keys())

        if kargs_dict is None:
            kargs_dict = {}
        for key in algo_dict.keys():
            if key not in kargs_dict:
                kargs_dict[key] = {}

        self.algo_dict, self.kargs_dict = algo_dict, kargs_dict
        self.num_exp = num_exp
        self.custom_metric_dict = custom_metric_dict
        self.__dict__.update(kargs) # absorb all the kargs arguments here. May be useful for custom methods specified by user.
        self.n = 1

    def bechmark_custom_dataset(self, dataset_list, discrete=False):
        '''
        This module helps evaluate the performance of one or more causal discovery algorithms on user provided data.

        :param dataset_list: The data must be a list of tuples, where each tuple contains the triplet (data_array, var_names, graph_gt), 
            where data_array is a 2D Numpy data array of shape (samples x variables), var_names is a list of variable names, 
            and graph_gt is the ground truth causal graph in the form of a Python dictionary, where keys are the variable names, 
            and the corresponding values are a list of parent names.
        :type dataset_list: List[Tuple]
        :param discrete: Specify if all the datasets contain discrete or continuous variables. This information is only used to 
            decide whether to standardize the data arrays or not. If discrete is False, all the data arrays are standardized.
        '''
        result_list = []
        self.variant_values = ['']
        self.variant_name = ''
        for i,(data_array, var_names, graph_gt) in enumerate(tqdm.tqdm(dataset_list)):
            assert data_array.shape[1]==len(var_names)==len(graph_gt.keys()), f'The number of columns in data_array, the length of '\
                            f'var_names and number of keys in graph_gt should be same but found {data_array.shape[1]}, '\
                            f'{len(var_names)} and {len(graph_gt.keys())}. This error occured at the {i}th index of dataset_list.'
            assert set(var_names)==set(graph_gt.keys()), f'The names of variables in variable name list and the keys of the '\
                            f'graph must match, but found {var_names} and {list(graph_gt.keys())}. This error occured at the {i}th index of dataset_list.'
            data_obj = _get_data_obj(data_array, var_names, discrete)
            result_dict = {}
            for algo_name, algo in self.algo_dict.items():
                tic = time.time()
                algo_ = algo(data=data_obj)
                result = algo_.run(**self.kargs_dict[algo_name])
                toc = time.time()
                graph_est = extract_graph_est(result)
                precision, recall, f1_score = get_precision_recall(graph_est, graph_gt)
                
                result_dict[algo_name] = {'time_taken': toc-tic,
                                          'precision': precision,
                                          'recall': recall,
                                          'f1_score': f1_score}

                for metric_name, metric_fn in self.custom_metric_dict.items():
                    metric_val = metric_fn(graph_est, graph_gt)
                    result_dict[algo_name][metric_name] = metric_val

                result_list.append(result_dict)
        self.results_full = [result_list]

    def aggregate_results(self, metric_name):
        '''
        This method aggregates the causal discovery results generated by one of the benchmarking methods (which must be run first), 
        and produces a result mean and a result standard deviation array. Both these arrays have shape (num_algorithms x num_variants), 
        where num_algorithms is the number of causal discovery algorithms specified in the benchmarking module, and num_variants is 
        the number of configurations of the argument being varied (e.g. in benchmark_variable_complexity, the number of variables 
        specified). Note that for the bechmark_custom_dataset, num_variants=1.

        :param metric_name: String specifying which metric (E.g. Precision) to aggregate from the generated results.
        :type metric_name: str
        '''
        variants_len = len(self.results_full)
        results_mean = np.zeros((len(self.algo_names), variants_len)) 
        results_sd = np.zeros((len(self.algo_names), variants_len))
        for i,algo_name in enumerate(self.algo_names):
            for j,results_list in enumerate(self.results_full):
                assert metric_name in results_list[0][algo_name].keys(), f'{metric_name} not found in the given result data. '\
                                        f'Feasible options are {list(results_list[0][algo_name].keys())}'
                results_this = [results_list[k][algo_name][metric_name] for k in range(len(results_list))]
                results_mean[i,j] = np.mean(results_this)
                results_sd[i,j] = np.std(results_this)
        self.results_mean = results_mean # shape: num_algorithms x num_variants
        self.results_std = results_sd # shape: num_algorithms x num_variants


    def plot(self, metric_name='f1_score', xaxis_mode=1):
        '''
        This method plots the aggregated results for `metric_name`. Y-axis is the metric_name, and x-axis can be one of two 
        things-- algorithm names, or the variant values, depending on the specified value of xaxis_mode.

        :param metric_name: String specifying which metric (E.g. Precision) to aggregate from the generated results.
        :type metric_name: str
        :param xaxis_mode: Integer (0 or 1) specifying what to plot on the x-axis. When 0, x-axis is algorithm names,
            and when 1, x-axis is the values of the variant. Variant denotes the configurations of the argument being 
            varied (e.g. in benchmark_variable_complexity, the number of variables).
        :type xaxis_mode: int
        '''
        if not hasattr(self, 'results_full'):
            print(f'Run a benchmark method before calling plot.')
            return

        valid_metric_names = ['precision', 'recall', 'f1_score', 'time_taken'] + list(self.custom_metric_dict.keys())
        assert metric_name in valid_metric_names,\
                    f"metric_name must be one of {valid_metric_names}, but give {metric_name}."
        self.aggregate_results(metric_name)
        plt = _plot(self.algo_names, self.variant_name, self.variant_values, self.results_mean, self.results_std,
                       metric_name, xaxis_mode)
        return plt

def extract_graph_est(result):
    graph_est={n:[] for n in result.keys()}
    for key in result.keys():
        parents = result[key]['parents']
        graph_est[key].extend(parents)
    return graph_est


def _get_data_obj(data_array, var_names, discrete):
    if discrete is False:
        StandardizeTransform_ = StandardizeTransform()
        StandardizeTransform_.fit(data_array)
        data_trans = StandardizeTransform_.transform(data_array)
    else:
        data_trans = data_array
    data_obj = TabularData(data_trans, var_names=var_names)
    return data_obj

def base_synthetic_tabular_benchmark(algo_dict, kargs_dict, noise_fn, num_vars=20, graph_density=0.1, T=1000, num_exp=20,\
                       fn:Callable = lambda x:x, coef=0.1, discrete=False, nstates=5, custom_metric_dict={}):
    result_list = []
    var_names = [str(i) for i in range(num_vars)]
    for seed in tqdm.tqdm(range(num_exp)):
        sem = GenerateSparseTabularSEM(var_names=var_names, graph_density=graph_density, seed=seed, fn=fn, coef=coef)
        data_array, var_names, graph_gt = DataGenerator(sem, T=T, seed=seed, discrete=discrete, nstates=nstates,\
                                            noise_fn=noise_fn)
        data_obj = _get_data_obj(data_array, var_names, discrete)
        result_dict = {}
        for algo_name, algo in algo_dict.items():
            tic = time.time()
            algo_ = algo(data=data_obj)
            result = algo_.run(**kargs_dict[algo_name])
            toc = time.time()
            graph_est = extract_graph_est(result)
            precision, recall, f1_score = get_precision_recall(graph_est, graph_gt)
            
            result_dict[algo_name] = {'time_taken': toc-tic,
                                      'precision': precision,
                                      'recall': recall,
                                      'f1_score': f1_score}

            for metric_name, metric_fn in custom_metric_dict.items():
                metric_val = metric_fn(graph_est, graph_gt)
                result_dict[algo_name][metric_name] = metric_val

        result_list.append(result_dict)
    return result_list

def _plot(algo_names, variant_name, variant_values, results_mean, results_std,
                       metric_name, xaxis_mode=1):
    assert xaxis_mode in [0,1], f"The argument xaxis_mode can only be one of 0 or 1, "\
                                f'but given {xaxis_mode}. 0: xaxis will have algorithms'\
                                f', and 1: xaxis will have the parameter being varied.'
    if xaxis_mode==0:
        for i in range(results_mean.shape[1]):
            plt.errorbar(algo_names, results_mean[:,i], yerr=results_std[:,i],
                         fmt='.-', linewidth=1, label=variant_values[i], capsize=3)
    elif xaxis_mode==1:
        for i in range(results_mean.shape[0]):
            plt.errorbar(variant_values, results_mean[i], yerr=results_std[i],
                         fmt='.-', linewidth=1, label=algo_names[i], capsize=3)

    plt.grid()
    plt.ylabel(metric_name)
    if xaxis_mode==1:
        plt.xlabel(variant_name)
    plt.legend()
    return plt

######


### Continuous data class ###
class BenchmarkContinuousTabularBase(BenchmarkTabularBase):
    '''
    Base class for the tabular data benchmarking module for the continuous case. This class inherits the methods and 
    variables from BenchmarkTabularBase, and defines 
    the dictionaries of default causal discovery algorithms and their default respective arguments.
    
    '''
    default_algo_dict = {
            'PC-PartialCorr':partial(PC, CI_test=PartialCorrelation(), use_multiprocessing=False,
                                      prior_knowledge=None),
            'GES':partial(GES, use_multiprocessing=False, prior_knowledge=None), 
            'LINGAM':partial(LINGAM, use_multiprocessing=False, prior_knowledge=None)}

    default_kargs_dict = {
            'PC-PartialCorr': {'max_condition_set_size': 4, 'pvalue_thres': 0.01},
            'GES': {'phases': ['forward', 'backward', 'turning']},
            'LINGAM': {'pvalue_thres': 0.01}}

    def __init__(self, algo_dict:Dict=None, kargs_dict:Dict=None, num_exp:int=20, custom_metric_dict:Optional[Dict]={}, **kargs):
        '''
         Benchmarking module for continuous tabular data.
        
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
        custom_metric_dict = {} if custom_metric_dict is None else custom_metric_dict
        if algo_dict is None:
            algo_dict = self.default_algo_dict

        if kargs_dict is None:
            kargs_dict = self.default_kargs_dict

        BenchmarkTabularBase.__init__(self, algo_dict=algo_dict, num_exp=num_exp, kargs_dict=kargs_dict,
                                        custom_metric_dict=custom_metric_dict, **kargs)

######


### Discrete data class ###

class BenchmarkDiscreteTabularBase(BenchmarkTabularBase):
    '''
    Base class for the tabular data benchmarking module for the discrete case. This class inherits the methods and 
    variables from BenchmarkTabularBase, and defines 
    the dictionaries of default causal discovery algorithms and their default respective arguments.
    '''
    default_algo_dict = {
            'PC-Pearson':partial(PC, CI_test=DiscreteCI_tests(method="pearson"), use_multiprocessing=False,
                                      prior_knowledge=None),
            'PC-Log-Likelihood':partial(PC, CI_test=DiscreteCI_tests(method="log-likelihood"), use_multiprocessing=False,
                                      prior_knowledge=None),
            'PC-Mod-Log-Likelihood':partial(PC, CI_test=DiscreteCI_tests(method="mod-log-likelihood"), use_multiprocessing=False,
                                      prior_knowledge=None),
            'PC-Freeman-Tukey':partial(PC, CI_test=DiscreteCI_tests(method="freeman-tukey"), use_multiprocessing=False,
                                      prior_knowledge=None),
            'PC-Neyman':partial(PC, CI_test=DiscreteCI_tests(method="neyman"), use_multiprocessing=False,
                                      prior_knowledge=None),}

    default_kargs_dict = {
            'PC-Pearson': {'max_condition_set_size': 4, 'pvalue_thres': 0.01},
            'PC-Log-Likelihood': {'max_condition_set_size': 4, 'pvalue_thres': 0.01},
            'PC-Mod-Log-Likelihood': {'max_condition_set_size': 4, 'pvalue_thres': 0.01},
            'PC-Freeman-Tukey': {'max_condition_set_size': 4, 'pvalue_thres': 0.01},
            'PC-Neyman': {'max_condition_set_size': 4, 'pvalue_thres': 0.01},}

    def __init__(self, algo_dict:Dict=None, kargs_dict:Dict=None, num_exp:int=20, custom_metric_dict:Optional[Dict]={}, **kargs):
        '''
        Benchmarking module for discrete tabular data.
        
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
        custom_metric_dict = {} if custom_metric_dict is None else custom_metric_dict
        if algo_dict is None:
            algo_dict = self.default_algo_dict

        if kargs_dict is None:
            kargs_dict = self.default_kargs_dict

        BenchmarkTabularBase.__init__(self, algo_dict=algo_dict, num_exp=num_exp, kargs_dict=kargs_dict,
                                        custom_metric_dict=custom_metric_dict, **kargs)

######
