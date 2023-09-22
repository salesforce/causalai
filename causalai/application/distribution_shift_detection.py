'''
TabularDistributionShiftDetector detects the origins of distribution shifts in tabular, continous/discrete data 
with the help of domain index variable. The algorithm uses the PC algorithm to estimate the causal graph,
by treating distribution shifts as intervention of the domain index on the root cause node, and PC can use 
conditional independence tests to quickly recover the causal graph and detec the root cause of anomaly.
Note that the algorithm supports both discrete and continuous variables, and can handle nonlinear relationships
by converting the continous variables into discrete ones using K-means clustering and using discrete PC algorithm
instead for CI test and causal discovery.

This algorithm makes the following assumptions: 
1. observational samples conditioned on the domain index are i.i.d. 
2. arbitrary relationship between variables,
3. Causal Markov condition, which implies that two variables that are d-separated in a causal graph are 
probabilistically independent
4. faithfulness, i.e., no conditional independence can hold unless the Causal Markov condition is met,
5. no hidden confounders. 
'''
import numpy as np
import pandas as pd
from typing import List, Union, Optional, Dict

from causalai.models.tabular.pc import PCSingle, PC
from causalai.models.common.CI_tests.discrete_ci_tests import DiscreteCI_tests
from causalai.data.data_generator import DataGenerator, GenerateRandomTabularSEM, _DiscretizeData
from causalai.data.tabular import TabularData
from causalai.data.transforms.tabular import StandardizeTransform, Heterogeneous2DiscreteTransform
from causalai.models.common.prior_knowledge import PriorKnowledge
from causalai.application.common import invert_graph_and_remove_duplicates

class TabularDistributionShiftDetector():
    '''
    Detects the root causes of distribution shift in tabular data.
    
    Reference: Ikram, Azam, et al. "Root Cause Analysis of Failures in 
    Microservices through Causal Discovery."
    Advances in Neural Information Processing Systems 35 (2022): 31158-31170.
    ''' 
    def __init__(
        self, 
        data_obj: TabularData,
        var_names: List[str],
        domain_index_name: str='domain_index', 
        prior_knowledge: Optional[PriorKnowledge]=None
    ):
        '''
        PC algorithm for root cause detection in domain-varying data settings.
        :param data_obj: tabular data object
        :type data_obj: TabularData
        :param var_names: list of variable names
        :type var_names: List[str]
        :param domain_index_name: name of the domain index column
        :tyoe domain_index_name: str
        :param prior_knowledge: prior knowledge about the causal graph
        :type prior_knowledge: Optional[PriorKnowledge]
        '''
        assert domain_index_name in var_names, 'Domain index not found in the data!'
        self.data_obj = data_obj
        self.var_names = var_names
        self.prior_knowledge = prior_knowledge
        self.domain_index_name = domain_index_name
    
    def run(
        self, 
        pvalue_thres: float=0.01, 
        max_condition_set_size: int=4,
        return_graph: bool=False
    ):
        '''
        Run the algorithm for root cause detection in tabular data.
        :param pvalue_thres: p-value threshold for conditional independence test
        :type pvalue_thres: float
        :param max_condition_set_size: maximum size of the condition set
        :type max_condition_set_size: int
        :return_graph: whether to return the estimated causal graph
        :type return_graph: bool
        :return: root cause of the incident and/or the estimated causal graph
        :rtype: Union[List[str], Dict[str, List[str]]]
        '''
        data_obj, var_names = self.data_obj, self.var_names
        # Forbid links that point from variables to the domain index
        if self.prior_knowledge is None:
            forbidden_links  = {self.domain_index_name: 
                [var_name for var_name in var_names if var_name != self.domain_index_name]}
            prior_knowledge = PriorKnowledge(forbidden_links=forbidden_links)
        else:
            prior_knowledge = self.prior_knowledge
        # Run discrete PC algorithm on the processed data
        CI_test = DiscreteCI_tests(method='pearson')
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
        root_causes = inv_map[self.domain_index_name]
        print(f'The distribution shifts are from the nodes: {root_causes}')
        if return_graph:
            return root_causes, inv_map
        else:
            return root_causes