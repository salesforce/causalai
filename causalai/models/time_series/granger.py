'''

Granger causality can be used for causal discovery in time series data without contemporaneous causal connections. The intuition behind 
Granger causality is that for two time series random variables X and Y, if including the past values of X to predict Y improves 
the prediction performance, over using only the past values of Y, then X causes Y. In practice, to find the causal parents of a 
variable, this algorithm involves performing linear regression to predict that variable using the remaining variables, and using the 
regression coefficients to determine the causality.

Granger causality assumes: 1. linear relationship between variables, 2. covariance stationary, i.e., a temporal sequence of random variables 
all have the same mean and the covariance between the random variables at any two time steps depends only on their relative positions, and 3. no hidden confounders.

Note that the Granger algorithm only supports lagged causal relationship discovery, i.e., no instantaneous causal relationships.
'''
from __future__ import print_function
from typing import Tuple, List, Union, Optional, Dict
from ...data.time_series import TimeSeriesData
from ...models.common.prior_knowledge import PriorKnowledge
from .base import BaseTimeSeriesAlgo, BaseTimeSeriesAlgoFull, ResultInfoTimeseriesSingle, ResultInfoTimeseriesFull
from numpy import ndarray
import warnings
import itertools
from collections import defaultdict
from copy import deepcopy
import numpy as np
import scipy.stats
import time
from sklearn.linear_model import LassoCV
import sys
import os
from scipy import stats
try:
    import ray
except:
    pass

class GrangerSingle(BaseTimeSeriesAlgo):
    def __init__(self, data: TimeSeriesData, prior_knowledge: Optional[PriorKnowledge]=None, max_iter: int= 1000,\
                    cv: int=5, use_multiprocessing: Optional[bool]=False):
        '''
        Granger causality algorithm for estimating lagged parents of single variable.

        :param data: this is a TimeSeriesData object and contains attributes likes data.data_arrays, which is a 
            list of numpy array of shape (observations N, variables D).
        :type data: TimeSeriesData object
        :param prior_knowledge: Specify prior knoweledge to the causal discovery process by either
            forbidding links that are known to not exist, or adding back links that do exist
            based on expert knowledge. See the PriorKnowledge class for more details.
        :type prior_knowledge: PriorKnowledge object
        :param max_iter:  max_iters to update the LassoCV least squares optimization (default=1000).
            See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html.
        :type max_iter: int
        :param cv: cross-validation generator or iterable (default=5).
            See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html.
        :type cv: int
        :param use_multiprocessing: If True, computations are performed using multi-processing which makes the algorithm faster.
        :type use_multiprocessing: bool
        '''
        BaseTimeSeriesAlgo.__init__(self, data=data, prior_knowledge=prior_knowledge, max_iter=max_iter, cv=cv, use_multiprocessing=use_multiprocessing)
        self.lasso_cv = LassoCV(max_iter=self.max_iter, cv=self.cv)
        self.get_correlation_and_pval = (lambda z,y: self._get_correlation_and_pval(z,y)) 
        if self.use_multiprocessing==True:
            if 'ray' in globals(): # Ray wrapper; avoiding Ray Actors because they are slower
                self.get_correlation_and_pval = ray.remote(self.get_correlation_and_pval)
            else:
                print('use_multiprocessing was specified as True but cannot be used because the ray library is not installed. Install using pip install ray.')

    def _get_correlation_and_pval(self, z: ndarray, y: ndarray) -> Tuple[Tuple[float,float], str]:
        '''
        Given lagged variables z and current time step variable y, this function computes the Granger regression
            coefficients and the corresponding p-values. This is a private function used internally by this class.
        '''
        msg = ''
        # get coefficients
        self.lasso_cv.fit(z, y)
        coef = self.lasso_cv.coef_

        # get p-value of each coefficient: see https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
        deg_freedom = z.shape[0]-z.shape[1]
        pred = np.dot(z, coef)
        MSE = (sum((y-pred)**2))/deg_freedom
        cov = np.dot(z.T,z)
        rank = np.linalg.matrix_rank(cov) # np.dot(z.T,z)
        if rank<z.shape[1]:
            msg = f'p-values cannot be calculated for the regression coefficients in Granger causality model because'\
                        f'the rank of the data matrix is too low ({rank}). Try increasing the number of samples.'
            pval = np.ones((cov.shape[0],)) # setting pvalues to 1 so no nodes are concluded as parents due to lack of data.
        else:
            standard_error = np.sqrt(MSE*(np.linalg.inv(cov).diagonal()))
            t_score = coef/ standard_error
            pval = stats.t.sf(np.abs(t_score), deg_freedom) * 2
            
        return list(zip(coef, pval)), msg

    def run(self, target_var: Union[int,str], pvalue_thres: float=0.05, max_lag: int=1, full_cd: bool=False) -> ResultInfoTimeseriesSingle:
        """
        Runs Granger causality algorithm for estimating the causal stength of all potential lagged parents of a single variable.

        :param target_var: Target variable index or name for which lagged parents need to be estimated.
        :type target_var: int or str
        :param pvalue_thres: Significance level used for hypothesis testing (default: 0.05). Candidate parents with pvalues above pvalue_thres
            are ignored, and the rest are returned as the cause of the target_var.
        :type pvalue_thres: float
        :param max_lag: Maximum time lag. Must be larger or equal to 1 (default: 1).
        :type max_lag: int, optional
        :param full_cd: This variable is only meant for internal use to handle multiprocessing if set to True (default: False).
        :type full_cd: bool

        :return: Dictionay has three keys:

            - parents : List of estimated parents.

            - value_dict : Dictionary of form {(var3_name, -1):float, ...} containing the test statistic of a link.

            - pvalue_dict : Dictionary of form {(var3_name, -1):float, ...} containing the
            p-value corresponding to the above test statistic.
        :rtype: dict
        """
        assert target_var in self.data.var_names, f'{target_var} not found in the variable names specified for the data!'
        self.target_var = target_var

        self.start(full_cd)

        candidate_parents = self.get_candidate_parents(target_var, max_lag)
        all_parents = self.get_all_parents(target_var, max_lag) 
        parents = deepcopy(candidate_parents)
        self.value_dict = {(p[0], p[1]): None for p in all_parents}
        self.pvalue_dict = {(p[0], p[1]): None for p in all_parents}

        if len(candidate_parents)<1:
            self.parents = self.get_parents(pvalue_thres)
            self.result = {'parents': self.parents,
                        'value_dict': self.value_dict,
                        'pvalue_dict': self.pvalue_dict,
                        'undirected_edges': []}
            return self.result

        val_pval_ray = []
        X = target_var
        Z = parents
        
        x,_,z = self.data.extract_array(X=X, Y=None, Z=Z, max_lag=max_lag) # x is time series length; z is time series length x num candidate dims

        # Perform independence test
        if self.use_multiprocessing==True and 'ray' in globals():
            val_pval_ray = self.get_correlation_and_pval.remote(z,x)
        else:
            val_pval_ray = self.get_correlation_and_pval(z,x)

        if self.use_multiprocessing==True and 'ray' in globals():
            val_pval_ray = ray.get(val_pval_ray)
        val_pval_ray, msg = (val_pval_ray)
        if msg!='': print(msg)
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


class Granger(BaseTimeSeriesAlgo, BaseTimeSeriesAlgoFull):
    '''
    Granger algorithm for estimating lagged parents of all variables.
    '''

    def __init__(self, data: TimeSeriesData, prior_knowledge: Optional[PriorKnowledge]=None, max_iter: int= 1000,\
                     cv: int=5, use_multiprocessing: Optional[bool]=False, **kargs):
        '''
        Granger causality algorithm for estimating lagged parents of all variables.

        :param data: this is a TimeSeriesData object and contains attributes likes data.data_arrays, which is a 
            list of numpy array of shape (observations N, variables D).
        :type data: TimeSeriesData object
        :param prior_knowledge: Specify prior knoweledge to the causal discovery process by either
            forbidding links that are known to not exist, or adding back links that do exist
            based on expert knowledge. See the PriorKnowledge class for more details.
        :type prior_knowledge: PriorKnowledge object
        :param max_iter:  max_iters to update the LassoCV least squares optimization (default=1000).
            See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html.
        :type max_iter: int
        :param cv: cross-validation generator or iterable (default=5).
            See https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html.
        :type cv: int
        :param use_multiprocessing: If True, computations are performed using multi-processing which makes the algorithm faster.
        :type use_multiprocessing: bool
        '''
        BaseTimeSeriesAlgo.__init__(self, data=data, prior_knowledge=prior_knowledge, max_iter=max_iter, cv=cv, use_multiprocessing=use_multiprocessing)
        BaseTimeSeriesAlgoFull.__init__(self, **kargs)

    def run(self, pvalue_thres: float=0.05, max_lag: int=1) -> Dict[Union[int,str],ResultInfoTimeseriesFull]:
        """
        Runs Granger causality algorithm for estimating the causal stength of all potential lagged parents
        of all the variables.

        :param pvalue_thres:  This pvalue_thres is the significance level used for hypothesis testing (default: 0.05).
        :type pvalue_thres: float
        :param max_lag: Maximum time lag. Must be larger or equal to 1 (default: 1).
        :type max_lag: int, optional 

        :return: Dictionay has D keys, where D is the number of variables. The value corresponding each key is 
            the dictionary output of GrangerSingle.run.
        :rtype: dict
        """
        self.start()
        for name in self.data.var_names:
            granger_d = GrangerSingle(self.data, prior_knowledge=self.prior_knowledge, cv=self.cv, use_multiprocessing=self.use_multiprocessing)
            result = granger_d.run(target_var=name, pvalue_thres=pvalue_thres, max_lag=max_lag, full_cd=True)
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

