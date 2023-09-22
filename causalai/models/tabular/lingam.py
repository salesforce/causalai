'''
LINGAM can be used for causal discovery in tabular data. The algorithm works by first performing independent component analysis (ICA) on the 
observational data matrix X (#variables x #samples) to extract the mixing matrix A over the independent components (noise matrix) E (same size as X), i.e. solving X=AE. 
Then their algorithm uses the insight that to find the causal order, each sample x can be decomposed as, x = Bx + e, where B is a lower triangular 
matrix and e are the independent noise samples. Noticing that B = (I - A^-1), we solve for B, and find the permutation matrix P, such that PBP' is 
as close to a lower triangular matrix as possible.

This algorithm makes the following assumptions: 1. linear relationship between variables, 2. non-Gaussianity of the error (regression residuals), 
3. The causal graph is a DAG, 4. no hidden confounders. We do not support multi-processing for this algorithm.
'''
from __future__ import print_function
from typing import TypedDict, Tuple, List, Union, Optional, Dict
import lingam
from lingam.utils import make_prior_knowledge
import warnings
import itertools
from collections import defaultdict
from copy import deepcopy
import numpy as np
from scipy import stats
from statsmodels.tsa.vector_ar.var_model import VAR
import scipy.stats
import time
import os
from ...data.tabular import TabularData
from ...models.common.prior_knowledge import PriorKnowledge
from .base import BaseTabularAlgo, BaseTabularAlgoFull, ResultInfoTabularSingle, ResultInfoTabularFull

class LINGAM(BaseTabularAlgo, BaseTabularAlgoFull):
    '''
    LiNGAM algorithm exploits the additive non-Gaussian residual in linear causal graphs for causal discovery
    on multivariate tabular data.

    References:
    [1] Shimizu, Shohei, Patrik O. Hoyer, Aapo Hyvärinen, Antti Kerminen, and Michael Jordan. 
    "A linear non-Gaussian acyclic model for causal discovery." Journal of Machine Learning Research 7, no. 10 (2006).
    '''

    def __init__(self, data: TabularData, prior_knowledge: Optional[PriorKnowledge]=None,
                            use_multiprocessing: Optional[bool]=False, **kargs):
        '''
        LiNGAM algorithm wrapper.
        
        :param data: this is a TabularData object and contains attributes likes data.data_arrays, which is a 
            list of numpy array of shape (observations N, variables D).
        :type data: TabularData object
        :param prior_knowledge: Specify prior knowledge to the causal discovery process by either
            forbidding links that are known to not exist, or adding back links that do exist
            based on expert knowledge. See the PriorKnowledge class for more details.
        :type prior_knowledge: PriorKnowledge object
        :param use_multiprocessing: Multi-processing is not supported.
        :type use_multiprocessing: bool
        '''

        #multi-processing for lingam has not implemented
        BaseTabularAlgo.__init__(self, data=data, prior_knowledge=prior_knowledge, use_multiprocessing=False)
        BaseTabularAlgoFull.__init__(self, **kargs)

    def run(self, pvalue_thres: float=0.05) -> Dict[Union[int,str],ResultInfoTabularFull]:
        """
        Runs LiNGAM algorithm for estimating the causal graph.
        
        :param pvalue_thres:  This pvalue_thres is the significance level used for hypothesis testing (default: 0.05).
        :type pvalue_thres: float

        :return: Dictionay has D keys, where D is the number of variables. The value corresponding to each key is 
            a dictionary with three keys:

            - parents : List of estimated parents.

            - value_dict : Dictionary of form {var3_name:float, ...} containing the test statistic of a link.

            - pvalue_dict : Dictionary of form {var3_name:float, ...} containing the p-value corresponding to the above test statistic.
        :rtype: dict
        """
        assert len(self.data.data_arrays)==1, f'LINGAM model can only accept TabularData '\
                        f' object with exactly 1 tabular array, but found {len(self.data.data_arrays)} arrays.'
                        
        pk_matrix = self.to_lingam_prior_knowledge(self.prior_knowledge, self.data.var_names)

        data_array_ret, = self.data.data_arrays
        
        model = lingam.DirectLiNGAM(prior_knowledge=pk_matrix)
        model.fit(data_array_ret)


        lingam_input = self.data.data_arrays[0]
        error_results = lingam_input - np.dot(lingam_input, model.adjacency_matrix_.T)

        all_parents = {}
        for var_idx in range(model.adjacency_matrix_.shape[0]):
            var = self.data.var_names[var_idx]
            value_dict = dict()
            for source_var_idx in range(model.adjacency_matrix_.shape[0]):
                parent = self.data.var_names[source_var_idx]
                value_dict[parent] = model.adjacency_matrix_[var_idx, source_var_idx]

            # compute pvalues    
            pvalue_dict = self._calculate_pvalues(var, value_dict, lingam_input, error_results)

            self.result[var] = {'value_dict': value_dict,
                                'pvalue_dict': pvalue_dict}

        all_parents = self.get_parents(pvalue_thres=pvalue_thres)
        for key in self.result.keys():
            self.result[key]['parents'] = all_parents[key]
        return self.result

    def _calculate_pvalues(self, var, value_dict, lingam_input, error_results):
        coef = list(value_dict.values())

        deg_freedom = error_results.shape[0] - len(coef)
        MSE = (error_results[:,self.data.var_names.index(var)]**2).sum()/deg_freedom
        
        z_names = self.data.var_names
        z = lingam_input
        cov = np.dot(z.T,z)
        
        rank = np.linalg.matrix_rank(cov)
        if rank<z.shape[1]:
            msg = f'p-values cannot be calculated for the regression coefficients in LINGAM model because'\
                        f'the rank of the data matrix is too low ({rank}). Try increasing the number of samples.'
            pval = np.ones((cov.shape[0],)) # setting pvalues to 1 so no nodes are concluded as parents due to lack of data.
        else:
            standard_error = np.sqrt(MSE*(np.linalg.inv(cov).diagonal()))
            t_score = coef/ standard_error
            pval = stats.t.sf(np.abs(t_score), deg_freedom) * 2

        pvalue_dict = {n: p for n,p in zip(z_names, pval)}
        return pvalue_dict

    def get_parents(self, pvalue_thres: float=0.05,
                    target_var: Optional[Union[int,str]]=None) -> Dict[Union[int,str], Tuple[Tuple[Union[int,str],int]]]:
        '''
        Assuming run() function has been called, get_parents function returns a dictionary. The keys of this
        dictionary are the variable names, and the corresponding values are the list of
        parent names that cause the target variable under the given pvalue_thres.

        :param pvalue_thres: This pvalue_thres is the significance level used for hypothesis testing (default: 0.05).
        :type pvalue_thres: float
        :param target_var: If specified (must be one of the data variable names), the parents of only this variable
            are returned as a list, otherwise a dictionary is returned where each key is a target variable
            name, and the corresponding values is the list of its parents.
        :type target_var: str or float, optional

        :return: Dictionay has D keys, where D is the number of variables. The value corresponding each key is 
            the list of parent names that cause the target variable under the given pvalue_thres.
        :rtype: dict
         '''
        parents = BaseTabularAlgoFull.get_parents(self, pvalue_thres=pvalue_thres, target_var=target_var)
        return parents

    def to_lingam_prior_knowledge(self, pk, var_names):
        forbidden_links = pk.forbidden_links
        existing_links = pk.existing_links
        root_variables = pk.root_variables
        leaf_variables = pk.leaf_variables

        paths = []
        no_paths = []
        if forbidden_links is not None:
            for key, vals in forbidden_links.items():
                for forbidden_parent in vals:
                    forb_par = var_names.index(forbidden_parent)
                    child = var_names.index(key)
                    no_paths.append((forb_par, child))
        
        for var in root_variables:
            for i in range(len(var_names)):
                if var!=var_names[i]:
                    no_paths.append((i, var_names.index(var)))
        
        if existing_links is not None:
            for key, vals in existing_links.items():
                for existing_parent in vals:
                    exist_par = var_names.index(existing_parent)
                    child = var_names.index(key)
                    paths.append((exist_par, child))
        
        sink_variables = [var_names.index(i) for i in leaf_variables]
        pk_matrix = make_prior_knowledge(n_variables=len(var_names),\
                                         sink_variables=sink_variables, paths=paths, no_paths=no_paths)
        return pk_matrix

