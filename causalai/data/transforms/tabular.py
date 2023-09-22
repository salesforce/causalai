from abc import abstractmethod
from collections import defaultdict, OrderedDict
from typing import Tuple, List, Union, Optional, Dict
import sys
import warnings
import copy
import math
import numpy as np
from numpy import ndarray
from sklearn.preprocessing import StandardScaler
from .base import BaseTransform, BaseHeterogeneous2DiscreteTransform


class StandardizeTransform(BaseTransform):
    '''
    Standardize tabular data by subtracting mean and dividing by standard deviation
    '''
    def __init__(self, with_mean: bool=True, with_std: bool=True):
        """
        :param with_mean: subtract mean from data if True
        :type with_mean: bool
        :param with_std: scale data by its standard deviation if True
        :type with_std: bool
        """
        super().__init__(with_mean=with_mean, with_std=with_std)

    def fit(self, *data: List[ndarray]) -> None:
        """
        Function that transforms the data arrays and stores any transformation parameter associated with the transform as a class attribute (i.e., mean, variance).
        StandardScaler ignores any NaN values along a column when computing column mean and standard deviation.

        :param data: Numpy array of shape (observations N, variables D)
        :type data: ndarray
        """
        self.transforms = [StandardScaler(with_mean=self.with_mean, with_std=self.with_std) for _ in range(len(data))]
        for i in range(len(data)):
            self.transforms[i].fit(data[i])

        total_n_samples_seen_ = sum([self.transforms[i].n_samples_seen_ for i in range(len(self.transforms))])
        self.global_mean, self.global_var = 0., 1.
        if self.with_mean:
            self.global_mean = sum([self.transforms[i].mean_* self.transforms[i].n_samples_seen_ for i in range(len(self.transforms))])/\
                                total_n_samples_seen_  
        if self.with_std:
            sec_mom = sum([ (self.transforms[i].var_ + self.transforms[i].mean_**2)* self.transforms[i].n_samples_seen_ for i in range(len(self.transforms)) ])/\
                                total_n_samples_seen_
            self.global_var = sec_mom - self.global_mean**2

    def transform(self, *data: List[ndarray]) -> Union[List[ndarray],ndarray]:
        """
        Function that returns the transformed data array list using the transform learned using the fit function

        :param data: Numpy array of shape (observations N, variables D)
        :type data: ndarray

        :return: transformed data
        :rtype: ndarray or list of ndarray
        """
        transformed_data = []
        for i in range(len(data)):
            new_data = (data[i] - self.global_mean)/np.sqrt(self.global_var + 1e-7)
            transformed_data.append(new_data)
        return transformed_data if len(transformed_data)>1 else transformed_data[0]

class Heterogeneous2DiscreteTransform(BaseHeterogeneous2DiscreteTransform):
    '''
    Transform heterogeneous data with mixed discrete and continuous variables to all discrete variables. Only continuous variables are trasformed to dicrete.
    '''
    def __init__(self, nstates: int=10):
        """
        :param nstates: nstates specifies the number of bins to use for discretizing
        :type nstates: int
        """
        super().__init__(nstates=nstates)

    def fit(self, *data: List[ndarray], var_names:List, discrete: Dict) -> None:
        """
        :param data: Numpy array of shape (observations N, variables D)
        :type data: ndarray
        :param var_names: List of variable names corresponding to the columns of the data array
        :type var_names: List
        :param discrete: The keys of this dictionary must be the variable names and the value corresponding to
            each key must be True or False. A value of False implies the variable is continuous, and discrete otherwise.
        :type discrete: dict
        """
        BaseHeterogeneous2DiscreteTransform.fit(self, *data, var_names=var_names, discrete=discrete)

    def transform(self, *data: List[ndarray]) -> Union[List[ndarray],ndarray]:
        """
        Function that returns the transformed data array list using the transform learned using the fit function

        :param data: Numpy array of shape (observations N, variables D)
        :type data: ndarray

        :return: transformed data
        :rtype: ndarray or list of ndarray
        """
        return BaseHeterogeneous2DiscreteTransform.transform(self, *data)

