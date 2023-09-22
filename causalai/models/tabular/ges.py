'''
Greedy Equivalence Search (GES) heuristically searches the space of causal Bayesian network and returns the model with highest 
Bayesian score it finds. Specifically, GES starts its search with the empty graph. It then performs a forward search in which
edges are added between nodes in order to increase the Bayesian score. This process
is repeated until no single edge addition increases the score. Finally, it performs a backward
search that removes edges until no single edge removal can increase the score.

This algorithm makes the following assumptions: 
1. observational samples are i.i.d. 
2. linear relationship between variables with Gaussian noise terms,
3. Causal Markov condition, which implies that two variables that are d-separated in a causal graph are 
probabilistically independent
4. faithfulness, i.e., no conditional independence can hold unless the Causal Markov condition is met,
5. no hidden confounders. 
We do not support multi-processing for this algorithm.
'''
from __future__ import print_function
from typing import TypedDict, Tuple, List, Union, Optional, Dict
import lingam
import warnings
import itertools
from collections import defaultdict
from copy import deepcopy
from numpy import ndarray
import numpy as np
from scipy import stats
import scipy.stats
import time
import os
from ...data.tabular import TabularData
from ...models.common.prior_knowledge import PriorKnowledge
from .base import BaseTabularAlgo, BaseTabularAlgoFull, ResultInfoTabularSingle, ResultInfoTabularFull
import ges

class GES(BaseTabularAlgo, BaseTabularAlgoFull):
    '''
    Greedy Equivalence Search (GES) for estimating the causal graph from multivariate tabular data. This class is a 
    wrapper around the GES library: https://github.com/juangamella/ges.
    library 

    Reference: Chickering, David Maxwell. "Optimal structure identification with greedy search." 
    Journal of machine learning research 3.Nov (2002): 507-554.
    
    '''

    def __init__(self, data: TabularData, prior_knowledge: Optional[PriorKnowledge]=None,
                        use_multiprocessing: Optional[bool]=False, **kargs):
        '''
        Greedy Equivalence Search (GES) for estimating the causal graph from tabular data.
        
        :param data: this is a TabularData object and contains attributes likes data.data_arrays, which is a 
            list of numpy array of shape (observations N, variables D).
        :type data: TabularData object
        :param prior_knowledge: Specify prior knoweledge to the causal discovery process by either
            forbidding links that are known to not exist, or adding back links that do exist
            based on expert knowledge. See the PriorKnowledge class for more details.
        :type prior_knowledge: PriorKnowledge object
        :param use_multiprocessing: Multi-processing is not supported.
        :type use_multiprocessing: bool
        '''

        # multi-processing for GES is not supported
        BaseTabularAlgo.__init__(self, data=data, prior_knowledge=prior_knowledge, use_multiprocessing=False)
        BaseTabularAlgoFull.__init__(self, **kargs)

    def run(self, pvalue_thres: Optional[float]=None, A0: Optional[ndarray] = None, phases: List[str] = ['forward', 'backward', 'turning'],\
                        debug: int = 0) -> Dict[Union[int,str],ResultInfoTabularFull]:
        """
        Runs GES algorithm for estimating the causal graph.
        
        :param pvalue_thres:  Ignored in this algorithm.
        :type pvalue_thres: float
        :param A0: the initial CPDAG on which GES will run, where where A0[i,j] != 0 implies i -> j and 
            A0[i,j] != 0 & A0[j,i] != 0 implies i - j. Defaults to the empty graph.
        :type A0: np.array
        :param phases:  this controls which phases of the GES procedure are run, and in which order. 
            Defaults to ['forward', 'backward', 'turning']. 
        :type phases: list[str]
        :param debug: if larger than 0, debug are traces printed. Higher values correspond to increased verbosity.
        :type debug: int, optional

        :return: Dictionay has D keys, where D is the number of variables. The value corresponding to each key is 
            a dictionary with three keys:

            - parents : List of estimated parents.

            - value_dict : Empty Python dictionary.

            - pvalue_dict : Empty Python dictionary.
        :rtype: dict
        """
        if pvalue_thres != None:
            print('Warning: pvalue_thres is not used in GES. Ignoring pvalue_thres.')
        assert len(self.data.data_arrays)==1, f'GES model can only accept TabularData '\
                        f' object with exactly 1 tabular array, but found {len(self.data.data_arrays)} arrays.'
                        
        estimate, score = ges.fit_bic(self.data.data_arrays[0], A0 = A0, phases = phases, debug = debug)
        
        graph = self._adjacency_matrix2graph(estimate)
        graph = self.prior_knowledge.post_process_tabular(graph)

        self.score = score # score achieved by the causal graph estimated using GES. Higher is better.

        for key in self.data.var_names:
            self.result[key] = {'value_dict': {}, 'pvalue_dict': {}}
            self.result[key]['parents'] = graph[key]
        return self.result
    
    def _adjacency_matrix2graph(self, M):
        '''
        M: m x m matrix, where rows are parents, and columns with 1 are children, 0 not children.
        '''
        names = self.data.var_names
        assert M.shape[0]==M.shape[1] and len(M.shape)==2
        names = names if names is not None else list(range(M.shape[0]))
        graph = {name:[] for name in names}
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i,j]==1:
                    graph[names[j]].append(names[i])
        return graph
    
