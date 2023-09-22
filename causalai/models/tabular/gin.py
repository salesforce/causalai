'''
Generalized Independent Noise (GIN) is a method for causal discovery for tabular data when there are 
hidden confounder variables.

Let X denote the set of all the observed variables and L the set of unknown ground truth hidden variables. 
Then this algorithm makes the following assumptions:
1. There is no observed variable in X, that is an ancestor of any latent variables in L.
2. The noise terms are non-Gaussian.
3. Each latent variable set L' in L, in which every latent variable directly causes the same set of 
observed variables, has at least 2Dim(L') pure measurement variables as children.
4. There is no direct edge between observed variables.
'''
from typing import TypedDict, Tuple, List, Union, Optional, Dict
from itertools import combinations
import warnings
import itertools
from collections import defaultdict
from copy import deepcopy
import numpy as np
from scipy.stats import chi2
import time
import os
from causalai.data.tabular import TabularData
from causalai.models.common.prior_knowledge import PriorKnowledge
from ...models.common.CI_tests.partial_correlation import PartialCorrelation
from ...models.common.CI_tests.kci import KCI
from ...models.common.CI_tests.discrete_ci_tests import DiscreteCI_tests
from causalai.models.tabular.base import BaseTabularAlgo, BaseTabularAlgoFull, ResultInfoTabularSingle, ResultInfoTabularFull
from causalai.models.common.CI_tests.kci import KCI
import warnings

class GIN(BaseTabularAlgo, BaseTabularAlgoFull):
    '''
    Generalized Independent Noise (GIN) is a method for causal discovery for multivariate tabular data when there are 
    hidden confounder variables.

    References:
    [1] Xie, F., Cai, R., Huang, B., Glymour, C., Hao, Z., & Zhang, K. (2020). Generalized independent noise condition 
    for estimating latent variable causal graphs. Advances in neural information processing systems, 33, 14891-14902.
    '''

    def __init__(self, data: TabularData, prior_knowledge: Optional[PriorKnowledge]=None,
                            CI_test: Union[PartialCorrelation,KCI,DiscreteCI_tests]=KCI(),
                            use_multiprocessing: Optional[bool]=False, **kargs):
        '''
        Generalized Independent Noise (GIN) is a method for causal discovery when there are hidden confounder variables.
        
        :param data: this is a TabularData object and contains attributes likes data.data_arrays, which is a 
            list of numpy array of shape (observations N, variables D).
        :type data: TabularData object
        :param prior_knowledge: Prior knowledge is not supported for the GIN algorithm.
        :type prior_knowledge: PriorKnowledge object
        :param use_multiprocessing: Multi-processing is not supported.
        :type use_multiprocessing: bool
        '''

        BaseTabularAlgo.__init__(self, data=data, prior_knowledge=prior_knowledge, use_multiprocessing=False)
        BaseTabularAlgoFull.__init__(self, **kargs)
        
        if prior_knowledge is not None:
            warnings.warn("Prior knowledge is not supported for the GIN algorithm, but a prior_knowledge object waas passed. Ignoring.")
        self.covariance =np.cov(data.data_arrays[0].T)

        self.check_indpendence = lambda x,y: self._check_independence(x,y)
        self.CI_test = CI_test#KCI()
        if self.use_multiprocessing==True and 'ray' in globals():
            self.check_indpendence = ray.remote(self.check_indpendence)

    def run(self, pvalue_thres: float=0.05) -> Dict[Union[int,str],ResultInfoTabularFull]:
        """
        Runs GIN algorithm for estimating the causal graph with latent variables.
        
        :param pvalue_thres:  This pvalue_thres is the significance level used for hypothesis testing (default: 0.05).
        :type pvalue_thres: float

        :return: Dictionay has D keys, where D is the number of variables. The value corresponding to each key is 
            a dictionary with three keys:

            - parents : List of estimated parents.

            - value_dict : Dictionary of form {var3_name:float, ...} containing the test statistic of a link.

            - pvalue_dict : Dictionary of form {var3_name:float, ...} containing the p-value corresponding to the above test statistic.
        :rtype: dict
        """
        assert len(self.data.data_arrays)==1, f'GIN model can only accept TabularData '\
                        f' object with exactly 1 tabular array, but found {len(self.data.data_arrays)} arrays.'
                        
        self.start()
        causal_cluster_list = []
        num_common_confounders = 1 # we start by looking for causal clusters where vars share exactly 1 confounder var; Len in paper
        unclustered_vars = set(range(len(self.data.var_names))) # here unclustered means vars for which the causal cluster has not been identified; P in paper

        # step 1: identify causal clusters
        test_results = []
        while num_common_confounders < len(unclustered_vars):
            cluster_list = []
            for var_list in combinations(unclustered_vars, num_common_confounders):
                var_list_bar = unclustered_vars - set(var_list)
                test_result = self.check_indpendence(var_list, var_list_bar)
                test_results.append(test_result)
            if self.use_multiprocessing==True and 'ray' in globals():
                test_results = ray.get(test_results)
            for (fisher_pval, fisher_stat, var_list) in test_results:
                if fisher_pval >= pvalue_thres:
                    cluster_list.append(var_list)
            cluster_list = merge_connected_components(cluster_list) # Merge all the overlapping sets
            causal_cluster_list += cluster_list
            for cluster in cluster_list:
                unclustered_vars -= set(cluster)
            num_common_confounders += 1

        self.causal_cluster_list = causal_cluster_list # mathcal{L} in algorithm 2 in paper

        # step 2: find the causal order of latent variables
        causal_order_kappa = [] # this variable corresponds to kappa in paper
        done = False
        while not done: 
            done = True
            Y_hat, Z_hat = [], []
            for T in causal_order_kappa:
                clusterT1, clusterT2 = segment_data(T, 2)
                Z_hat = Z_hat + clusterT1
                Y_hat = Y_hat + clusterT2

            for r, L_Sr in enumerate(causal_cluster_list):
                L_Sr_root = True # assume the latent vars of L_Sr are root unless we find a violation
                L_Sr1, L_Sr2 = segment_data(L_Sr, 2)
                for k, L_Sk in enumerate(causal_cluster_list):
                    if k == r:
                        continue
                    L_Sk1, _ = segment_data(L_Sk, 2)
                    fisher_pval, fisher_stat, _= self._check_independence(L_Sr1 + L_Sk1 + Z_hat, L_Sr2 + Y_hat)
                    if fisher_pval < pvalue_thres:
                        L_Sr_root = False
                        break
                if L_Sr_root:
                    causal_cluster_list.remove(L_Sr)
                    causal_order_kappa.append(L_Sr)
                    done = False
                    break

        causal_cluster_list_ = []
        for cluster in causal_cluster_list:
            cluster = [self.data.var_names[i] for i in cluster]
            causal_cluster_list_.append(cluster)
        causal_cluster_list = causal_cluster_list_

        causal_order_kappa_ = []
        for cluster in causal_order_kappa:
            cluster = [self.data.var_names[i] for i in cluster]
            causal_order_kappa_.append(cluster)
        causal_order_kappa = causal_order_kappa_

        # create the estimated graph using the above results
        def add_node(var_name):
            if var_name not in self.result:
                self.result[var_name] = {'value_dict': {},
                                    'pvalue_dict': {},
                                    'parents': []}
        def add_edge(parent, child):
            self.result[child]['parents'].append(parent)

        causal_graph = {}
        for var in unclustered_vars:
            add_node(var)

        latent_var_num = 0
        latent_vars = []

        for cluster in causal_order_kappa:
            latent_var = f'L{latent_var_num}'
            add_node(latent_var)
            for var in latent_vars:
                add_edge(var, latent_var)
            latent_vars.append(latent_var)

            for var in cluster:
                add_node(var)
                add_edge(latent_var, var)
            latent_var_num += 1

        undirected_latent_nodes = []

        for cluster in causal_cluster_list:
            latent_var = f'L{latent_var_num}'
            add_node(latent_var)
            for var in latent_vars:
                add_edge(var, latent_var)

            for var in undirected_latent_nodes:
                add_edge(var, latent_var)
                add_edge(latent_var, var)

            undirected_latent_nodes.append(latent_var)

            for var in cluster:
                add_node(var)
                add_edge(latent_var, var)
            latent_var_num += 1

        self.stop()
        self.causal_order = causal_order_kappa
        return self.result

    def _check_independence(self, var_list1, var_list2):
        data_array = self.data.data_arrays[0]
        var_idx1 = list(var_list1) 
        var_idx2 = list(var_list2)
        E_Y_Z = get_E_Y_Z(data_array, self.covariance, var_idx1, var_idx2)
        fisher_stat = 0.
        for z in range(len(var_list2)):
            p_k = self.CI_test.run_test(data_array[:, [z]], E_Y_Z[:, None])[1]
            p_k = max(p_k, 1e-7)
            fisher_stat += -2.* np.log(p_k)
        fisher_pval = 1. - chi2.cdf(fisher_stat, 2 * len(var_list2))
        return fisher_pval, fisher_stat, var_list1
    

def merge_connected_components(clusters):
    adjacency_list = defaultdict(set)
    for var_list in clusters:
        for i in range(len(var_list)):
            for j in range(i+1, len(var_list)):
                adjacency_list[var_list[i]].add(var_list[j])
                adjacency_list[var_list[j]].add(var_list[i])
    
    def dfs(node, connected_component):
        visited.add(node)
        connected_component.append(node)
        for neigh in adjacency_list[node]:
            if neigh not in visited:
                dfs(neigh, connected_component)
    
    connected_components = []
    visited = set()
    for node in adjacency_list:
        if node not in visited:
            connected_component = []
            dfs(node, connected_component)
            connected_components.append(connected_component)
    return connected_components


def segment_data(x, n_segments):
    segments = []
    idx = 0
    segment_len = len(x) // n_segments
    extra_segments = len(x) % n_segments
    for i in range(extra_segments):
        segments.append(x[idx:idx + segment_len + 1])
        idx = idx + segment_len + 1

    for i in range(n_segments - extra_segments):
        segments.append(x[idx:idx + segment_len])
        idx = idx + segment_len
    return segments

def get_E_Y_Z(data, cov, X, Z):
    cov_m = cov[np.ix_(Z, X)]
    _, _, v = np.linalg.svd(cov_m)
    omega = v.T[:, -1]
    return np.dot(data[:, X], omega)
