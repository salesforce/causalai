'''
VARLINGAM can be used for causal discovery in time series data with contemporaneous causal connections. This algorithm can be broadly 
divided into two steps. First, we estimate the time lagged causal effects using vector autoregression. Second, we estimate the instantaneous 
causal effects by applying the LiNGAM algorithm on the residuals of the previous step, where LiNGAM exploits the non-Gaussianity of the 
residuals to estimate the instantaneous variables' causal order.

This algorithm makes the following assumptions: 1. linear relationship between variables, 2. non-Gaussianity of the error (regression residuals), 
3. no cycles among contemporaneous causal relations, and 4. no hidden confounders. We do not support multi-processing for this algorithm.
'''
from __future__ import print_function
from typing import TypedDict, Tuple, List, Union, Optional, Dict
import lingam
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
from ...models.common.CI_tests.partial_correlation import PartialCorrelation
from ...models.common.CI_tests.kci import KCI
from ...data.time_series import TimeSeriesData
from .base import BaseTimeSeriesAlgo, BaseTimeSeriesAlgoFull, ResultInfoTimeseriesFull

class VARLINGAM(BaseTimeSeriesAlgo, BaseTimeSeriesAlgoFull):
    '''
    VAR-LiNGAM algorithm which combines non-Gaussian instantanenous model with autoregressive model for causal discovery
    on multivariate time series data

    References:
    [1] Aapo Hyvärinen, Kun Zhang, Shohei Shimizu, Patrik O. Hoyer.
    Estimation of a Structural Vector Autoregression Model Using Non-Gaussianity.
    Journal of Machine Learning Research, 11: 1709-1731, 2010.
    '''

    def __init__(self, data: TimeSeriesData, use_multiprocessing: Optional[bool]=False, **kargs):
        '''
        VAR-LiNGAM algorithm wrapper.
        
        :param data: this is a TimeSeriesData object and contains attributes likes data.data_arrays, which is a 
            list of numpy array of shape (observations N, variables D).
        :type data: TimeSeriesData object
        :param use_multiprocessing: Multi-processing is not supported.
        :type use_multiprocessing: bool
        '''

        #multi-processing for var-lingam has not implemented
        BaseTimeSeriesAlgo.__init__(self, data=data, prior_knowledge=None, use_multiprocessing=False)
        BaseTimeSeriesAlgoFull.__init__(self, **kargs)

    def run(self, pvalue_thres: float=0.05, max_lag: int=1) -> Dict[Union[int,str],ResultInfoTimeseriesFull]:
        """
        Runs VAR-LiNGAM algorithm for estimating the causal stength of all potential time-lagged and instantanenous
        causal variables.
        
        :param pvalue_thres:  This pvalue_thres is the significance level used for hypothesis testing (default: 0.05).
        :type pvalue_thres: float
        :param max_lag: Maximum time lag. Must be larger or equal to 1 (default: 1).
        :type max_lag: int, optional 

        :return: Dictionay has D keys, where D is the number of variables. The value corresponding each key is 
            a dictionary with three keys:

            - parents : List of estimated parents.

            - value_dict : Dictionary of form {(var3_name, -1):float, ...} containing the test statistic of a link.

            - pvalue_dict : Dictionary of form {(var3_name, -1):float, ...} containing the p-value corresponding to the above test statistic.
        :rtype: dict
        """
        assert len(self.data.data_arrays)==1, f'VARLINGAM model can only accept TimeSeriesData '\
                        f' object with exactly 1 time series array, but found {len(self.data.data_arrays)} arrays.'
        data_array_ret, = self.data.data_arrays
        model = lingam.VARLiNGAM(max_lag)
        model.fit(data_array_ret)



        all_parents = {}
        for var_idx in range(model.adjacency_matrices_.shape[1]):
            var = self.data.var_names[var_idx]
            value_dict = dict()
            for lags in range(model.adjacency_matrices_.shape[0]):
                for source_var_idx in range(model.adjacency_matrices_.shape[2]):
                    lag_name = -lags if lags > 0 else 0
                    parent = (self.data.var_names[source_var_idx], lag_name)
                    value_dict[parent] = model.adjacency_matrices_[lags, var_idx, source_var_idx]


            
            # compute pvalues    
            lingam_input = model._residuals
            error_results = lingam_input - np.dot(lingam_input, model.adjacency_matrices_[0].T)
            pvalue_dict = self._calculate_pvalues(var, value_dict, lingam_input, error_results)



            self.result[var] = {'value_dict': value_dict,
                                'pvalue_dict': pvalue_dict}

        all_parents = self.get_parents(pvalue_thres=pvalue_thres)
        for key in self.result.keys():
            self.result[key]['parents'] = all_parents[key]
        return self.result

    def _calculate_pvalues(self, var, value_dict, lingam_input, error_results):
        coef = list(value_dict.values())
        # irrespective of the max_lag value provided to VARLINGAM, it only outputs coefficients
        # of variables with max lag that it finds are relevant. So we find the max_lag_used
        # by VARLINGAM below before proceeding
        max_lag_used = int(len(coef)/len(self.data.var_names)) - 1

        deg_freedom = error_results.shape[0] - len(coef)
        MSE = (error_results[:,self.data.var_names.index(var)]**2).sum()/deg_freedom
        
        z_names = [(n,-t) for t in range(1,max_lag_used+1) for n in self.data.var_names]
        _,_,lagged_z = self.data.extract_array(X=var, Y=None,Z= z_names, max_lag=max_lag_used)
        l = min(lingam_input.shape[0], lagged_z.shape[0])
        z = np.concatenate([lingam_input[-l:], lagged_z[-l:]], axis=1)
        cov = np.dot(z.T,z)
        
        rank = np.linalg.matrix_rank(cov)
        if rank<z.shape[1]:
            msg = f'p-values cannot be calculated for the regression coefficients in VARLINGAM model because'\
                        f'the rank of the data matrix is too low ({rank}). Try increasing the number of samples.'
            pval = np.ones((cov.shape[0],)) # setting pvalues to 1 so no nodes are concluded as parents due to lack of data.
        else:
            standard_error = np.sqrt(MSE*(np.linalg.inv(cov).diagonal()))
            t_score = coef/ standard_error
            pval = stats.t.sf(np.abs(t_score), deg_freedom) * 2

        z_names = [(var, 0) for var in self.data.var_names] + z_names
        pvalue_dict = {n: p for n,p in zip(z_names, pval)}
        return pvalue_dict

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
        :type target_var: str or float, optional

        :return: Dictionay has D keys, where D is the number of variables. The value corresponding each key is 
            the list of parent names that cause the target variable under the given pvalue_thres.
        :rtype: dict
         '''
        parents = BaseTimeSeriesAlgoFull.get_parents(self, pvalue_thres=pvalue_thres, target_var=target_var)
        return parents

