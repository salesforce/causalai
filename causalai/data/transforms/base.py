from abc import abstractmethod
from collections import defaultdict, OrderedDict
from typing import Tuple, List, Union, Optional, Dict
import sys
import warnings
import copy
import math
import numpy as np
from numpy import ndarray
from ..data_generator import _DiscretizeData

class BaseTransform:
    '''
    Common Base Transform class for both time series data and tabular data
    '''
    def __init__(self, **kwargs):
    	self.__dict__.update(kwargs)

    @abstractmethod
    def fit(self, *data: List[ndarray]):
        """
        Function that transforms the data arrays and stores any transformation parameter associated with the transform as a class attribute (E.g. mean, variance)

        :param data: Numpy array of shape (observations N, variables D)
        :type data: ndarray
        """
        raise NotImplementedError()

    @abstractmethod
    def transform(self, *data: Union[List[ndarray],ndarray]):
        """
        Function that returns the transformed data array list using the transform learned using the fit function

        :param data: Numpy array of shape (observations N, variables D)
        :type data: ndarray

        :return: transformed data
        :rtype: ndarray or list of ndarray
        """
        raise NotImplementedError()


class BaseHeterogeneous2DiscreteTransform:
    '''
    Transform heterogeneous data with mixed discrete and continuous variables to all discrete variables. Only continuous variables are trasformed to dicrete.
    '''
    def __init__(self, nstates: int=10):
        """
        :param nstates: nstates specifies the number of bins to use for discretizing
        :type nstates: int
        """
        self.nstates = nstates

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
        data_all = np.concatenate(data, axis=0)
        discrete = [discrete[name] for name in var_names]
        discrete_idx = [i for i in range(len(discrete)) if discrete[i]==1]
        
        self.continuous_idx = idx = [i for i in range(len(discrete)) if discrete[i]==0]
        data_c = data_all[:, idx]
        
        self.DiscretizeData_ = _DiscretizeData()
        _=self.DiscretizeData_(data_c, data_c, self.nstates)
        self.discrete = discrete
        

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
            data_new = copy.deepcopy(data[i])
            for j in self.continuous_idx:
                data_new_j = self.DiscretizeData_.transform(data[i][:,j].reshape(-1,1),\
                                                            self.continuous_idx.index(j)).reshape(-1)
                data_new[:,j] = data_new_j
            transformed_data.append(data_new)
        return transformed_data if len(transformed_data)>1 else transformed_data[0]
    



