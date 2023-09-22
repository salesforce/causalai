'''
RootCauseDetector detects root cause of anomaly in continous time series data with 
the help of a higher-level context variable. The algorithm uses the PC algorithm to estimate the causal graph,
for root  cause analysis by treating the failure, represented using the higher-level metrics, as an intervention 
on the root cause node, and PC can use conditional independence tests to quickly detect 
which lower-level metric the failure node points to, as the root cause of anomaly.

This algorithm makes the following assumptions: 
1. observational samples conditioned on the higher-level context variable (e.g., time index) are i.i.d. 
2. linear relationship between variables with Gaussian noise terms,
3. Causal Markov condition, which implies that two variables that are d-separated in a causal graph are 
probabilistically independent
4. faithfulness, i.e., no conditional independence can hold unless the Causal Markov condition is met,
5. no hidden confounders. 
'''
import pandas as pd
import numpy as np
from typing import List, Union, Optional, Dict

from causalai.models.tabular.pc import PC
from causalai.data.tabular import TabularData
from causalai.models.common.prior_knowledge import PriorKnowledge
from causalai.models.common.CI_tests.partial_correlation import PartialCorrelation
from causalai.data.transforms.time_series import StandardizeTransform
from causalai.application.common import invert_graph_and_remove_duplicates

class RootCauseDetector():
    '''
    Detects root cause of distribution shift in time series data. 
    
    Reference: Ikram, Azam, et al. "Root Cause Analysis of Failures in Microservices through Causal Discovery."
    Advances in Neural Information Processing Systems 35 (2022): 31158-31170.
    
    Reference: Huang, Biwei, et al. "Causal discovery from heterogeneous/nonstationary data." 
    The Journal of Machine Learning Research 21.1 (2020): 3482-3534.
    '''
    def __init__(
        self, 
        data_obj: TabularData,
        var_names: List[str],
        time_metric_name: str='time', 
        prior_knowledge: Optional[PriorKnowledge]=None
    ):
        '''
        PC algorithm for root cause detection in time-varying data settings.
        :param data_obj: pre-processed TabularData object
        :type data_obj: TabularData
        :param var_names: list of variable names
        :type var_names: List[str]
        :param time_metric_name: name of the metric that represents time-varying context (e.g. time index)
        :type time_metric_name: str
            Defaults to the name 'time'.
        :param prior_knowledge: prior knowledge about the causal graph
        :type prior_knowledge: Optional[PriorKnowledge]
        '''
        assert time_metric_name in var_names, 'Time metric not found in the data!'
        self.data_obj = data_obj
        self.var_names = var_names
        self.time_metric_name = time_metric_name
        self.prior_knowledge = prior_knowledge
    
    def run(
        self, 
        pvalue_thres: float=0.05,
        max_condition_set_size: int=4,
        return_graph: bool=False
    ):
        '''
        Run the PC algorithm for root cause detection in microservice metrics.
        :param pvalue_thres: p-value threshold for conditional independence test
        :type pvalue_thres: float
            Defaults to 0.05.
        :param max_condition_set_size: maximum size of the condition set
        :type max_condition_set_size: int
            Defaults to 4.
        :param return_graph: whether to return the estimated causal graph
        :type return_graph: bool
            Defaults to False.
        :return: root cause of the incident and/or the estimated causal graph
        :rtype: Union[List[str], Dict[str, List[str]]]
        '''
        data_obj, var_names = self.data_obj, self.var_names
        # Forbid links that point from other metrics to the time metric
        if self.prior_knowledge is None:
            forbidden_links  = {self.time_metric_name: 
                [var_name for var_name in var_names if var_name != self.time_metric_name]}
            prior_knowledge = PriorKnowledge(forbidden_links=forbidden_links)
        else:
            prior_knowledge = self.prior_knowledge
        # Run PC algorithm on the processed data
        CI_test = PartialCorrelation()
        pc = PC(
            data=data_obj,
            prior_knowledge=prior_knowledge,
            CI_test=CI_test,
            use_multiprocessing=False
            )
        result = pc.run(
            pvalue_thres=pvalue_thres, 
            max_condition_set_size=max_condition_set_size
            )
        # Postprocess the PC algorithm result
        graph_est={n:[] for n in result.keys()}
        for key in result.keys():
            parents = result[key]['parents']
            graph_est[key].extend(parents)
        inv_map = invert_graph_and_remove_duplicates(graph_est)
        root_causes = inv_map[self.time_metric_name]
        print(f'The root cause(s) of the incident are: {root_causes}')
        if return_graph:
            return root_causes, inv_map
        else:
            return root_causes