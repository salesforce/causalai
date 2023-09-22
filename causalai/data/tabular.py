
from collections import defaultdict, OrderedDict
from numpy import ndarray
from typing import Tuple, List, Union, Optional
import sys
import warnings
import copy
import math
import numpy as np
import itertools
from .base import BaseData

class TabularData(BaseData):
    '''
    Data object containing tabular array.
    '''
    def __init__(self, data: ndarray, var_names: Optional[List[str]] = None, contains_nans: bool=False):
        """
        :param data: data is a Numpy array of shape (observations N, variables D).
        :type data: ndarray
        :param var_names: Names of variables. If None, range(N) is used.
        :type var_names: list
        :param contains_nans: If true, NaNs will be handled automatically during causal discovery. Note that
            checking for NaNs makes the code a little slower. So set to true only if needed.
        :type contains_nans: bool
        """
        super().__init__(data, var_names=var_names, contains_nans=contains_nans)

    def extract_array(self, X: Union[int,str], Y: Union[int,str], Z: List[Union[int,str]]) -> List[ndarray]:
        """
        Extract the arrays corresponding to the node names X,Y,Z from self.data_arrays (see BaseData). 
        X and Y are individual nodes, and Z is the set of nodes to be used as the
        conditional set.

        :param X: X is the target variable index/name. Eg. 3 or <var_name>, if a variable 
            name was specified when creating the data object.
        :type X: int or str
        :param Y: Y specifies a variable. Eg. 2 or <var_name>, if a variable 
            name was specified when creating the data object.
        :type Y: int or str
        :param Z: Z is a list of str or int, where each element has the form 2 or <var_name>, if a variable 
            name was specified when creating the data object.
        :type Z: list of str or int

        :return: x_array, y_array, z_array : Tuple of data arrays. All have 0th dimension equal to the number of 
            observarions. z_array.shape[1] has dimensions equal to the number of nodes specified in Z.
        :rtype: tuple of ndarray
        """

        X = [X]
        Y = [Y] if Y is not None else []



        X, Y, Z = self.to_var_index(X, Y, Z)

        XYZ = X + Y + Z
        total_num_nodes = len(XYZ)
        # XYZ = list(itertools.chain.from_iterable(XYZ))

        # Ensure that XYZ makes sense
        self.sanity_check(X,Y,Z, total_num_nodes)

        array = self.data_arrays[0][:, XYZ].T
        # remove all instances where NaN occurs
        if self.contains_nans:
            isnan = np.isnan(array)
            if isnan.any():
                idx = np.argwhere(isnan)[:,1]
                idx = [i for i in range(array.shape[1]) if i not in list(idx)]
                array = array[:,idx]

        x_array = array[0].reshape(-1)
        y_array = array[1].reshape(-1) if Y!=[] else None
        if Y!=[]:
            if Z!=[]:
                z_array = array[2:].T
            else:
                z_array = None
        else:
            if Z!=[]:
                z_array = array[1:].T
            else:
                z_array = None
        return x_array, y_array, z_array

    def to_var_index(self, *args) -> Union[List[List[Union[str, int]]], List[Union[str, int]]]:
        '''
        Convert variable names from string to variable index if the name is specified as a string.
        '''
        new_args = []
        for a in args:
            a_new = []
            if a != []:
                a_new = [self.var_name2index(ai) for ai in a] # list(map(self.var_name2index, a)) # 
            new_args.append(a_new)
        return new_args if len(new_args)>1 else new_args[0]

    def sanity_check(self, X: List[Tuple], Y: Union[List,List[Tuple]], Z: List[Tuple], total_num_nodes: int) -> None:
        """
        Perform the following checks:

        - The variable indices are between 0-D-1

        - There are no duplicate entries

        - Time lags are negative

        - Tuples have length 2 (index and time lag)
        
        :param X: list
        :param Y: list
        :param Z: list
        :param total_num_nodes: total number of nodes
        :type total_num_nodes: int
        """
        if (X[0] >= self.D or X[0] < 0):
                raise ValueError(f"Target variable X must be between 0-{self.D-1}. Found:\n{X}")

        for node_list in [Y,Z]:
            if node_list==[]:
                continue
            if len(node_list)!=len(set(node_list)):
                raise ValueError(f'Found duplicate entries in the following node list:\n{node_list}')
            # if len((node_list))!=1:
            #     raise ValueError(f"Node list must be a list of int or str of format [var index,...]. Eg., [2, 1, ...], but found\n{node_list}")
            if (np.any(np.array(node_list) >= self.D) or np.any(np.array(node_list) < 0)):
                raise ValueError(f"Node list must have variable indices between 0-{self.D-1}. Found:\n{node_list}")

    def get_causal_Xy(self, target_var: Union[int,str], parents: Tuple[Union[int, str]]) -> Tuple[ndarray, ndarray, List[Union[int,str]]]:
        '''
        Given target_var name, and the list of parents corresponding to target_var, this method
        extracts the data tuple of the form (X,y), where y is a 1D ndarray containing the observations corresponding to target_var
        as targets, and X is a 2D ndarray (num_observations, num_vars) where each row contains the variables in data that
        correspond to the parents of target_var. This pair (X,y) can be useful (for instance) for learning machine learning models 
        where X will be the input and y target.

        :param target_var: Target variable index or name.
        :type target_var: int
        :param parents: List of estimated parents of the form [<var5_name>, <var2_name>, ...].
        :type parents: list

        :return: X,y, column_names. X,y are as described above, and column_names is a list of names of the columns in X.
        :rtype: tuple(ndarray, ndarray, List)
        '''
        X_new = []
        column_names = parents
        if parents is not None and parents!=[]:
            for data_i in self.data_arrays:
                y_cor = [self.var_name2index(name) for name in parents] # y coordinate
                x = data_i[:, y_cor]
                X_new.append(x)
            X_new = np.vstack(X_new)
        else:
            X_new = None
        
        Y_new = np.stack([data_i[:,self.var_name2index(target_var)] for data_i in self.data_arrays])
        return X_new, Y_new.reshape(-1), column_names

    def get_causal_Xy_i(self, i:int, arr_idx:int, target_var: Union[int,str], parents: Tuple[Union[int, str]])\
                                     -> Tuple[ndarray, ndarray, List[Union[int,str]]]:
        '''
        Given target_var name, and the list of parents corresponding to target_var, this method
        extracts the data tuple of the form (X,y), where y is a 1 scalar containing the observation corresponding to target_var at index i
        as targets, and X is a 1D ndarray (1, num_vars) where the row contains the variables in data that
        correspond to the parents of target_var. This pair (X,y) can be useful (for instance) for prediction in machine learning models 
        where X will be the input and y target.

        :param i: row index of the data_array for which the target observation and its corresponding input needs to be extracted
        :type i: int
        :param arr_idx: index of the array in self.data_arrays
        :type arr_idx: int
        :param target_var: Target variable index or name.
        :type target_var: int
        :param parents: List of estimated parents of the form [<var5_name>, <var2_name>, ...].
        :type parents: list

        :return: X,y, column_names. X,y are as described above, and column_names is a list of names of the columns in X.
        :rtype: tuple(ndarray, ndarray, List)
        '''
        if arr_idx>=len(self.data_arrays):
            raise ValueError(f'Argument arr_idx to TabularData class method get_causal_Xy_i() was provided as {arr_idx} but must be less than {len(self.data_arrays)}.')
        if i>=self.data_arrays[arr_idx].shape[0]:
            raise ValueError(f'Argument i to TabularData class method get_causal_Xy_i() was provided as {i} but must be less than {self.data_arrays[arr_idx].shape[0]}.')
        data_i = self.data_arrays[arr_idx]

        X_new = []
        column_names = parents
        y_cor = [self.data.var_name2index(name) for name in parents] # y coordinate
        if parents is not None and parents!=[]:
            x = data_i[i, y_cor]
            X_new.append(x)
            X_new = np.stack(X_new)
        else:
            X_new = None
            return None, None
        
        Y_new = data_i[i,data.var_name2index(target_var)]
        return X_new, Y_new

