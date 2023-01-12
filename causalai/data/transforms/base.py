from abc import abstractmethod
from collections import defaultdict, OrderedDict
from typing import Tuple, List, Union, Optional
import sys
import warnings
import copy
import math
import numpy as np
from numpy import ndarray


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



