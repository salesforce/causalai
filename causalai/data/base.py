from abc import abstractmethod
from collections import defaultdict, OrderedDict
from typing import Tuple, List, Union, Optional
import sys
import warnings
import copy
import math
import numpy as np
from numpy import ndarray


class BaseData:
    '''
    Data object for tabular or time series array.
    '''
    def __init__(self, *data: List[ndarray], var_names: Optional[List[str]] = None, **kargs):
        """
        :param data: Each ndarray is a Numpy array of shape (observations N, variables D). In the case of
            time serie data, allowing multiple ndarray allows the user to pass multiple disjoint 
            time series (e.g. first series is data from Jan-March, while the second series is
            from July to September).
        :type data: list[ndarray]
        :param var_names: Names of variables. If None, range(N) is used.
        :type var_names: list
        """
        self.__dict__.update(kargs)
        self.data_arrays = [data_i.copy() for data_i in data]
        assert np.all([self.data_arrays[i].shape[1]==self.data_arrays[0].shape[1] for i in range(len(self.data_arrays))]),\
                                f'All the data arrays must have the same number of variables, but found {[a.shape[1] for a in self.data_arrays]}'
        D = data[0].shape[1] # D: number of variables/dimensions
        N = [data_i.shape[0] for data_i in self.data_arrays] # N: list of length of time series or number of observation
        # Set the variable names
        self.var_names = var_names
        # Set the default variable names if none are set
        if self.var_names is None:
            self.var_names = list(range(D))
        else:
            if any([type(s)!=str for s in self.var_names]):
                raise ValueError(f'Variable names must be strings. Found:\n {self.var_names}')
            if len(self.var_names) != D:
                raise ValueError("len(var_names) != data.shape[1].")

        self.N = N
        self.D = D

    @property
    def length(self) -> List[int]:
        '''
        Returns the list of length (0th dimensions) of each data array passed to the  constructor.
        '''
        return [data_i.shape[0] for data_i in self.data_arrays]
    @property
    def dim(self) -> int:
        '''
        Returns the number of variables (1st dimension) of the data arrays (which must be the same)
        '''
        return self.data_arrays[0].shape[1]

    def var_name2index(self, name: Union[int,str]):
        '''
        Convert variable names from strings to indices
        '''
        if type(name)==int:
            return name
        elif type(name)==str and name in self.var_names:
            return self.var_names.index(name)
        else:
            raise ValueError(f'Variable name {name} not found in the data object with specified names {self.var_names}. Use one of specified names or variable index.')

    def index2var_name(self, index: Union[int,str]):
        '''
        Convert indices to variable names string
        '''
        if type(index)==str:
            return index
        elif type(index)==int and index<len(self.var_names):
            return self.var_names[index]
        else:
            if type(index)==int and index>=len(self.var_names):
                raise ValueError(f'Variable index {index} must be less than {self.var_names} (the number of variables).')
            else:
                raise ValueError(f'Variable index {index} resulted in an error.')


    def var_name2array(self, var_name):
        return [arr[:, self.var_name2index(var_name)] for arr in self.data_arrays]

    @abstractmethod
    def extract_array(self, X: int, Y: Union[Tuple[Union[int,str], int], Union[int,str]], Z: Union[List[Tuple],List], max_lag: Optional[int]=None):
        """
        Extract the arrays corresponding to the node names X,Y,Z from self.data_arrays (see BaseData). 
        X and Y are individual nodes, and Z is the set of nodes to be used as the conditional set.

        :param X: X is the target variable at the current time step. Eg. 3 or <var_name>, if a variable 
            name was specified when creating the data object.
        :type X: int
        :param Y: Y specifies a variable. For tabular data it can be the variable index or name. For time series,
            it is the variable index/name at a specific time lag. Eg. (2,-1) or (<var_name>, -1), if a variable 
            name was specified when creating the data object. Here the time lag -1 implies it is 1 time
            step before X. The time lag must be negative. This is because: 1. a parent of Y cannot be at 
            a future time step relative to Y. 2. We do not support instantaneous causal links. Y can also be None.
        :type Y: tuple or int or str
        :param Z: For time series, Z is a list of tuples, where each tuple has the form (2,-1) or (<var_name>, -1), if 
            a variable name was specified when creating the data object. The time lags must be negative (same 
            reason as that specified for Y above). For tabular data, Z is a list of either variable indices or
            variable names.
        :type Z: list of tuples or a list
        :param max_lag: Maximum time lag from current time step specifying the Markov blanket lies within this interval.
        :type max_lag: int

        :return: x_array, y_array, z_array : Tuple of data arrays. All have 0th dimension equal to the length of time
            series. z_array.shape[1] has dimensions equal to the number of nodes specified in Z.
        :rtype: tuple of ndarray
        """

        raise NotImplementedError()