'''
The Peter-Clark (PC) algorithm is one of the most general purpose algorithms for causal discovery that can be used for both tabular and time series data, 
of both continuous and discrete types. Briefly, the PC algorithm works in two steps, it first identifies the undirected causal graph, and then (partially) 
directs the edges. In the first step, we check for the existence of a causal connection between every pair of variables by checking if there exists a condition 
set (a subset of variables excluding the two said variables), conditioned on which, the two variables are independent. In the second step, the edges are directed 
by identifying colliders. Note that the edge orientation strategy of the PC algorithm may result in partially directed graphs. In the case of time series data, 
the additional information about the time steps associated with each variable can also be used to direct the edges.

The PC algorithm makes four core assumptions: 1. Causal Markov condition, which implies that two variables that are d-separated in a causal graph are 
probabilistically independent, 2. faithfulness, i.e., no conditional independence can hold unless the Causal Markov condition is met, 3. no hidden 
confounders, and 4. no cycles in the causal graph. For time series data, it makes the additional assumption of stationarity, i.e., the properties of 
a random variable is agnostic to the time step.

Our implementation of the PC algorithm for time series currently only supports lagged causal relationship discovery, i.e., no instantaneous causal relationships.
'''
from __future__ import print_function
from typing import TypedDict, Tuple, List, Union, Optional, Dict
import warnings
import itertools
from collections import defaultdict
from copy import deepcopy
import numpy as np
import scipy.stats
import time
import os
from ...models.common.CI_tests.partial_correlation import PartialCorrelation
from ...models.common.CI_tests.kci import KCI
from ...models.common.CI_tests.discrete_ci_tests import DiscreteCI_tests
from ...data.time_series import TimeSeriesData
from ...models.common.prior_knowledge import PriorKnowledge
from .base import BaseTimeSeriesAlgo, BaseTimeSeriesAlgoFull, ResultInfoTimeseriesSingle, ResultInfoTimeseriesFull
try:
    import ray
except:
    pass

class GreedyConfigInfo(TypedDict):
    depth: int
    pvalue_thres: float

def _greedy_subroutine(data, CI_test, X, Y, Z_all, max_lag, condition_set_size, pvalue_thres):
    value, pvalue_max = None, None
    for i,Z in enumerate(itertools.combinations(Z_all, condition_set_size)):
        x,y,z = data.extract_array(X=X, Y=Y, Z=list(Z), max_lag=max_lag)
        # Perform independence test
        val, p_val = CI_test.run_test(x,y,z)
        if pvalue_max is None or pvalue_max<p_val:
            pvalue_max = p_val
            value = val
        if pvalue_max>pvalue_thres:
            return value, pvalue_max
    return value, pvalue_max

class PCSingle(BaseTimeSeriesAlgo):
    '''
    Peter-Clark (PC) algorithm for estimating lagged parents of single variable.
    '''

    def __init__(self, data: TimeSeriesData, prior_knowledge: Optional[PriorKnowledge]=None, 
                 CI_test: Union[PartialCorrelation,KCI,DiscreteCI_tests]=PartialCorrelation(), use_multiprocessing: Optional[bool]=False):
        '''
        PC algorithm for estimating lagged parents of single variable.

        :param data: this is a TimeSeriesData object and contains attributes likes data.data_arrays, which is a 
            list of numpy array of shape (observations N, variables D).
        :type data: TimeSeriesData object
        :param prior_knowledge: Specify prior knoweledge to the causal discovery process by either
            forbidding links that are known to not exist, or adding back links that do exist
            based on expert knowledge. See the PriorKnowledge class for more details.
        :type prior_knowledge: PriorKnowledge object
        :param CI_test: This object perform conditional independence tests (default: PartialCorrelation). 
            See object class for more details.
        :type CI_test: PartialCorrelation or KCI object
        :param use_multiprocessing: If True, computations are performed using multi-processing which makes the algorithm faster.
        :type use_multiprocessing: bool

        '''
        BaseTimeSeriesAlgo.__init__(self, data=data, prior_knowledge=prior_knowledge, CI_test=CI_test, use_multiprocessing=use_multiprocessing)

        self.CI_test_ = lambda x,y,z: CI_test.run_test(x,y,z)
        if use_multiprocessing:
            if 'ray' in globals():
                self.CI_test_ = ray.remote(self.CI_test_) # Ray wrapper; avoiding Ray Actors because they are slower
            else:
                print('use_multiprocessing was specified as True but cannot be used because the ray library is not installed. Install using pip install ray.')

    def run_greedy(self, target_var: Union[int,str], max_lag: int=1, depth: int=0,\
                        pvalue_thres: float=0.05, candidate_parents: Optional[List[Tuple]] = None) -> List[Tuple]: 
        if candidate_parents is None:
            candidate_parents = self.get_candidate_parents(target_var, max_lag)
        parents = deepcopy(candidate_parents) # subset of all_parents after ignoring links forbidden by prior_knowledge (if specified)
        all_parents = self.get_all_parents(target_var, max_lag)
        value_dict = {p: None for p in all_parents}
        pvalue_dict = {p: None for p in all_parents}

        depth = min(depth, len(all_parents)-1)# the max possible condition set size is len(all_parents)-1
        X = target_var

        greedy_subroutine = _greedy_subroutine
        if self.use_multiprocessing==True and 'ray' in globals():
            greedy_subroutine = ray.remote(greedy_subroutine)

        # global_nonsignificant_parents = set()
        for condition_set_size in range(depth+1):
            nonsignificant_parents = []
            val_pval_ray = []
            if len(all_parents)-1<condition_set_size:
                break

            for index_parent, parent in enumerate(parents):  
                Y = parent
                Z_all = [p for p in all_parents if p != parent]
                if self.use_multiprocessing==True and 'ray' in globals():
                    val_pval = greedy_subroutine.remote(self.data, self.CI_test, X, Y, Z_all, max_lag, condition_set_size, pvalue_thres)
                else:
                    val_pval = greedy_subroutine(self.data, self.CI_test, X, Y, Z_all, max_lag, condition_set_size, pvalue_thres)
                val_pval_ray.append(val_pval)

            if self.use_multiprocessing==True and 'ray' in globals():
                val_pval_ray = [ray.get(val_pval) for val_pval in val_pval_ray]

            val_ray = [v[0] for v in val_pval_ray]
            pval_ray = [v[1] for v in val_pval_ray]

            for index_parent, parent in enumerate(parents):
                if pval_ray[index_parent]>pvalue_thres: # no causal link b/w target and parent for the considered condition set
                    nonsignificant_parents.append(parent)
                if pvalue_dict[parent] is None or pvalue_dict[parent] < pval_ray[index_parent]:
                    # update pvalue_dict to store the largest pvalue seen (closer to being an non-significant parent)
                    pvalue_dict[parent] = pval_ray[index_parent]
                    value_dict[parent] = val_ray[index_parent]

            for parent in nonsignificant_parents:
                parents.remove(parent)
                all_parents.remove(parent)
        return parents, value_dict, pvalue_dict

    def run(self, target_var: Union[int,str], pvalue_thres: float=0.05, max_lag: int=1,\
                            max_condition_set_size: Optional[int] = 4, full_cd: bool=False) -> ResultInfoTimeseriesSingle:
        """
        Runs PC algorithm for estimating the causal stength of all potential lagged parents of a single variable.

        :param target_var: Target variable index or name for which lagged parents need to be estimated.
        :type target_var: int or str
        :param pvalue_thres: Significance level used for hypothesis testing (default: 0.05). Candidate parents with pvalues above pvalue_thres
            are ignored, and the rest are returned as the cause of the target_var.
        :type pvalue_thres: float
        :param max_lag: Maximum time lag. Must be larger or equal to 1 (default: 1).
        :type max_lag: int, optional
        :param max_condition_set_size:  If not None, independence tests using condition sets of 
            size {0,1,...max_condition_set_size} are performed
            (which are cheaper) before using condition sets involving all the candidate parents (default: 4).
            For example, max_condition_set_size = 0 implies that the greedy procedure 
            will only consider condition sets of size 0 to eliminate causal links between the target_var
            and a specific variable, if the pvalue between them turns out to be larger than pvalue_thres=0.05.
            Similarly max_condition_set_size=1 will consider condition sets of size 0 and 1. 
            The value of max_condition_set_size can be at maximum the total number of parents-1. If a value larger 
            than this is specified, max_condition_set_size is chosen as min(max_condition_set_size, len(all_parents)-1).
            If None is given, then condition sets involving all the candidate parents are used. While each CI test in 
            this case becomes more expensive than the greedy case, the number of CI tests in this cases is limited to 
            the number of candidate parents, which is less than the greedy case.
        :type max_condition_set_size: int
        :param full_cd: This variable is only meant for internal use to handle multiprocessing if set to True (default: False).
        :type full_cd: bool

        :return: Dictionay has three keys:

            - parents : List of estimated parents.

            - value_dict : Dictionary of form {(var3_name, -1):float, ...} containing the test statistic of a link.

            - pvalue_dict : Dictionary of form {(var3_name, -1):float, ...} containing the p-value corresponding to the above test statistic.
        :rtype: dict
        """

        assert target_var in self.data.var_names, f'{target_var} not found in the variable names specified for the data!'
        self.target_var = target_var
        self.start(full_cd)

        candidate_parents = self.get_candidate_parents(target_var, max_lag)
        all_parents = self.get_all_parents(target_var, max_lag) 
        if max_condition_set_size is not None:
            candidate_parents, value_dict, pvalue_dict = self.run_greedy(target_var, max_lag,\
                                            max_condition_set_size, pvalue_thres, candidate_parents)
            if max_condition_set_size>=len(all_parents)-1:
                self.value_dict = value_dict
                self.pvalue_dict = pvalue_dict
                self.parents = self.get_parents(pvalue_thres)
                self.stop()
                return {'parents': self.parents,
                        'value_dict': self.value_dict,
                        'pvalue_dict': self.pvalue_dict,
                        'undirected_edges': []}
        parents = deepcopy(candidate_parents)
        value_dict = {(p[0], p[1]): None for p in all_parents}
        pvalue_dict = {(p[0], p[1]): None for p in all_parents}
        
        # the reason for the need for all_parents in adddition to candidate_parents is that even if a variable may not be a causal parent, it may be needed
        # in the condition set to correctly determine other causal links.

        
        val_pval_ray = []
        X = target_var
        for index_parent, parent in enumerate(parents):            
            Y = parent
            Z = [p for p in all_parents if p != parent]
            
            x,y,z = self.data.extract_array(X=X, Y=Y, Z=Z, max_lag=max_lag)

            # Perform independence test
            if self.use_multiprocessing==True and 'ray' in globals():
                val_p_val = self.CI_test_.remote(x,y,z)
            else:
                val_p_val = self.CI_test_(x,y,z)
            val_pval_ray.append(val_p_val)

        if self.use_multiprocessing==True and 'ray' in globals():
            val_pval_ray = ray.get(val_pval_ray)

        val_ray = [v[0] for v in val_pval_ray]
        pval_ray = [v[1] for v in val_pval_ray]

        for index_parent, parent in enumerate(parents):
            self.pvalue_dict[parent] = pval_ray[index_parent]
            self.value_dict[parent] = val_ray[index_parent]

        self.parents = self.get_parents(pvalue_thres)
        self.stop(full_cd)
        self.result = {'parents': self.parents,
                    'value_dict': self.value_dict,
                    'pvalue_dict': self.pvalue_dict,
                    'undirected_edges': []}
        return self.result

class PC(BaseTimeSeriesAlgo, BaseTimeSeriesAlgoFull):
    '''
    Peter-Clark (PC) algorithm for estimating lagged parents of all variables.
    '''

    def __init__(self, data: TimeSeriesData, prior_knowledge: Optional[PriorKnowledge]=None, 
                 CI_test: Union[PartialCorrelation,KCI,DiscreteCI_tests]=PartialCorrelation(), use_multiprocessing: Optional[bool]=False, **kargs):
        '''
        PC algorithm for estimating lagged parents of all variables.
        
        :param data: this is a TimeSeriesData object and contains attributes likes data.data_arrays, which is a 
            list of numpy array of shape (observations N, variables D).
        :type data: TimeSeriesData object
        :param prior_knowledge: Specify prior knoweledge to the causal discovery process by either
            forbidding links that are known to not exist, or adding back links that do exist
            based on expert knowledge. See the PriorKnowledge class for more details.
        :type prior_knowledge: PriorKnowledge object
        :param CI_test: This object perform conditional independence tests (default: PartialCorrelation). 
            See object class for more details.
        :type CI_test: PartialCorrelation or KCI object
        :param use_multiprocessing: If True, computations are performed using multi-processing which makes the algorithm faster.
        :type use_multiprocessing: bool
        '''
        BaseTimeSeriesAlgo.__init__(self, data=data, prior_knowledge=prior_knowledge, CI_test=CI_test, use_multiprocessing=use_multiprocessing)
        BaseTimeSeriesAlgoFull.__init__(self, **kargs)

    def run(self, pvalue_thres: float=0.05, max_lag: int=1, max_condition_set_size: Optional[int] = None) -> Dict[Union[int,str],ResultInfoTimeseriesFull]:
        """
        Runs PC algorithm for estimating the causal stength of all potential lagged parents of all the variables.

        :param pvalue_thres: Significance level used for hypothesis testing (default: 0.05). Candidate parents with pvalues above pvalue_thres
            are ignored, and the rest are returned as the cause of the target_var.
        :type pvalue_thres: float
        :param max_lag: Maximum time lag. Must be larger or equal to 1 (default: 1).
        :type max_lag: int, optional
        :param max_condition_set_size:  If not None, independence tests using condition sets of 
            size {0,1,...max_condition_set_size} are performed
            (which are cheaper) before using condition sets involving all the candidate parents (default: 4).
            For example, max_condition_set_size = 0 implies that the greedy procedure 
            will only consider condition sets of size 0 to eliminate causal links between the target_var
            and a specific variable, if the pvalue between them turns out to be larger than pvalue_thres=0.05.
            Similarly max_condition_set_size=1 will consider condition sets of size 0 and 1. 
            The value of max_condition_set_size can be at maximum the total number of parents-1. If a value larger 
            than this is specified, max_condition_set_size is chosen as min(max_condition_set_size, len(all_parents)-1).
            If None is given, then condition sets involving all the candidate parents are used. While each CI test in 
            this case becomes more expensive than the greedy case, the number of CI tests in this cases is limited to 
            the number of candidate parents, which is less than the greedy case.
        :type max_condition_set_size: int

        :return: Dictionay has D keys, where D is the number of variables. The value corresponding each key is 
            the dictionary output of PCSingle.run.
        :rtype: dict
        """
        self.start()
        for name in self.data.var_names:
            pc_d = PCSingle(self.data, prior_knowledge=self.prior_knowledge, CI_test=self.CI_test, use_multiprocessing=self.use_multiprocessing)
            result = pc_d.run(target_var=name, pvalue_thres=pvalue_thres,max_lag=max_lag,\
                                             max_condition_set_size=max_condition_set_size, full_cd=True)
            result['parents'].extend(result['undirected_edges'])
            del result['undirected_edges']
            self.result[name] = result

        self.stop()
        return self.result

    def get_parents(self, pvalue_thres: float=0.05, 
                    target_var: Optional[Union[int,str]]=None) -> Dict[Union[int,str], Tuple[Tuple[Union[int,str],int]]]:
        '''
        Assuming run() function has been called, get_parents function returns a dictionary. The keys of this
        dictionary are the variable names, and the corresponding values are the list of 
        lagged parent names that cause the target variable under the given pvalue_thres.

        :param pvalue_thres: This pvalue_thres is the significance level used for hypothesis testing (default: 0.05).
        :type pvalue_thres: float
        :param target_var: If specified (must be one of the data variable names), the parents of only this variable
            are returned as a list, otherwise a dictionary is returned where each key is a target variable
            name, and the corresponding values is the list of its parents.
        :type target_var: str or int, optional

        :return: Dictionay has D keys, where D is the number of variables. The value corresponding each key is 
            the list of lagged parent names that cause the target variable under the given pvalue_thres.
        :rtype: dict
        '''
        parents = BaseTimeSeriesAlgoFull.get_parents(self, pvalue_thres=pvalue_thres, target_var=target_var)
        return parents

