
from collections import defaultdict, OrderedDict
from numpy import ndarray
from typing import Tuple, List, Union, Optional
import sys
import warnings
import copy
import math
import numpy as np
from .base import BaseData

class TimeSeriesData(BaseData):
    '''
    Data object containing time series array.
    '''
    def __init__(self, *data: List[ndarray], var_names: Optional[List[str]] = None, contains_nans: bool=False):
        """
        :param data: Each ndarray is a Numpy array of shape (observations N, variables D). This allows 
            the user to pass multiple disjoint time series (e.g. first series is data from 
            Jan-March, while the second series is from July to September).
        :type data: list of ndarray
        :param var_names:  Names of variables. If None, range(N) is used.
        :type var_names: list
        :param contains_nans: If true, NaNs will be handled automatically during causal discovery. Note that
            checking for NaNs makes the code a little slower. So set to true only if needed.
        :type contains_nans: bool
        """
        super().__init__(*data, var_names=var_names, contains_nans=contains_nans)

    def extract_array(self, X: Union[int,str], Y: Tuple[Union[int,str], int], Z: List[Tuple], max_lag: int) -> List[ndarray]:
        """
        Extract the arrays corresponding to the node names X,Y,Z from self.data_arrays (see BaseData). 
        X and Y are individual nodes, and Z is the set of nodes to be used as the
        conditional set.

        :param X: X is the target variable index/name at the current time step. Eg. 3 or <var_name>, if a variable 
            name was specified when creating the data object.
        :type X: int or str
        :param Y: Y specifies a variable at a specific time lag. Eg. (2,-1) or (<var_name>, -1), if a variable 
            name was specified when creating the data object. Here the time lag -1 implies it is 1 time
            step before X. The time lag must be negative. This is because: 1. a parent of Y cannot be at 
            a future time step relative to Y. 2. We do not support instantaneous causal links. Y can also be None.
        :type Y: tuple
        :param Z: Z is a list of tuples, where each tuple has the form (2,-1) or (<var_name>, -1), if a variable 
            name was specified when creating the data object. The time lags must be negative (same reason
            as that specified for Y above).
        :type Z: list of tuples
        :param max_lag: Maximum time lag from current time step specifying the Markov blanket lies within this interval.
        :type max_lag: int

        :return: x_array, y_array, z_array : Tuple of data arrays. All have 0th dimension equal to the total length of time
                series. z_array.shape[1] has dimensions equal to the number of nodes specified in Z.
        :rtype: tuple of ndarray
        """

        Z_refined = []
        for Zi in Z:
            if Zi[1]<-max_lag:
                continue
            Z_refined.append(Zi)
        Z = Z_refined

        X = [(X,0)]
        Y = [Y] if Y is not None else []



        X, Y, Z = self.to_var_index(X, Y, Z)


        XYZ = X + Y + Z
        total_num_nodes = len(XYZ)

        # Ensure that XYZ makes sense
        self.sanity_check(X,Y,Z, total_num_nodes)

        

        # Setup and fill array with lagged time series
        data_length = (np.array(self.N) - max_lag).sum()
        array = np.zeros((total_num_nodes, data_length), dtype=self.data_arrays[0].dtype)
        # Note, lags can only be non-positive

        
        for i, (var, lag) in enumerate(XYZ):
            l = []
            ctr=0
            for j,data_array in enumerate(self.data_arrays):
                if data_array.shape[0]>max_lag: # length of time series must be larger than the specified max_lag, otherwise ignore this series
                    array_j = data_array[max_lag + lag:self.N[j] + lag, var].reshape(-1)
                    array[i, ctr:ctr+len(array_j)] = array_j
                    ctr += len(array_j)

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

    def to_var_index(self, *args) -> Union[List[List[Tuple[Union[str,int],int]]], List[Tuple[Union[str,int],int]]]:
        '''
        Convert variable names from string to variable index if the name is specified as a string.
        '''
        new_args = []
        for a in args:
            a_new = []
            if a != []:
                a_new = [(self.var_name2index(ai[0]), ai[1]) for ai in a]
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
        if (np.any(np.array(X)[:, 0] >= self.D) or np.any(np.array(X)[:, 0] < 0)):
                raise ValueError(f"Target variable X must be between 0-{self.D-1}. Found:\n{X}")

        for node_list in [Y,Z]:
            if node_list==[]:
                continue
            if len(node_list)!=len(set(node_list)):
                raise ValueError(f'Found duplicate entries in the following node list:\n{node_list}')
            if len(np.array(node_list).shape)!=2 or np.array(node_list).shape[1] != 2:
                raise ValueError(f"Node list must be a list of tuples of format [(var index, -lag),...]. Eg., [(2, -2), (1, 0), ...], but found\n{node_list}")

            if np.any(np.array(node_list)[:, 1] > 0):
                raise ValueError(f'The following node list has positive lags. Positive lags violate causality because a future state cannot cause a past state.\n{node_list}')
            # if np.any(np.array(node_list)[:, 1] == 0):
            #     raise ValueError(f'The following node list has zero lags. Instantaneous causal links are not supported currently by our library.\n{node_list}')
            if (np.any(np.array(node_list)[:, 0] >= self.D) or np.any(np.array(node_list)[:, 0] < 0)):
                raise ValueError(f"Node list must have variable indices between 0-{self.D-1}. Found:\n{node_list}")

    def get_causal_Xy(self, target_var: Union[int,str], parents: Tuple[Tuple[Union[int, str],int]]) -> Tuple[ndarray, ndarray, List[Union[int,str]]]:
        '''
        Given target_var name, and the list of parents corresponding to target_var, this method
        extracts the data tuple of the form (X,y), where y is a 1D ndarray containing the observations corresponding to target_var
        as targets, and X is a 2D ndarray (num_observations, num_vars) where each row contains the variables in data that
        correspond to the parents of target_var. This pair (X,y) can be useful (for instance) for learning machine learning models 
        where X will be the input and y target.

        :param target_var: Target variable index or name.
        :type target_var: int
        :param parents: List of estimated parents of the form [(<var5_name>, -1), (<var2_name>, -3), ...].
        :type parents: list

        :return: X,y, column_names. X,y are as described above, and column_names is a list of names of the columns in X.
        :rtype: tuple(ndarray, ndarray, List)
        '''
        maxlag = -min([p[1] for p in parents]) if len(parents)>0 else None
        if maxlag is None:
            return [None]*3
        column_names = [name for (name,_) in parents] # y coordinate
        y_cor = [self.var_name2index(name) for (name,_) in parents] # y coordinate
        X_new = []
        if parents is not None and parents!=[]:
            for data_i in self.data_arrays:
                for t in range(maxlag, data_i.shape[0]):
                    x_cor = [t+j for (_,j) in parents] # x coordinate
                    xt = data_i[x_cor, y_cor]
                    X_new.append(xt)
            X_new = np.stack(X_new)
        else:
            X_new = None
        
        Y_new = np.stack([data_i[:,self.var_name2index(target_var)][maxlag:] for data_i in self.data_arrays])
        return X_new, Y_new.reshape(-1), column_names

    def get_causal_Xy_i(self, i:int, arr_idx:int, target_var: Union[int,str],\
                            parents: Tuple[Tuple[Union[int, str],int]])-> Tuple[ndarray, ndarray, List[Union[int,str]]]:
        '''
        Given a time series data object, target_var name, and the list of parents corresponding to target_var, this method
        extracts the data tuple of the form (X,y), where y is a scalar containing the observation corresponding to target_var at index i
        as targets, and X is a 1D ndarray where the row contains the variables in data that correspond to the parents of target_var.
        This pair (X,y) can be useful (for instance) for prediction in machine learning models where X will be the input and y target.

        :param i: row index of the data_array for which the target observation and its corresponding input needs to be extracted
        :type i: int
        :param arr_idx: index of the array in self.data_arrays
        :type arr_idx: int
        :param data: It contains the list data.data_arrays, where each item is a numpy array of shape (observations N, variables D).
        :type data: TimeSeriesData object
        :param target_var: Target variable index or name.
        :type target_var: int
        :param parents: List of estimated parents of the form [(<var5_name>, -1), (<var2_name>, -3), ...].
        :type parents: list

        :return: X,y as described above.
        :rtype: tuple(ndarray, ndarray)
        '''
        t = i
        if arr_idx>=len(self.data_arrays):
            raise ValueError(f'Argument arr_idx to TimeSeriesData class method get_causal_Xy_i() was provided as {arr_idx} but must be less than {len(self.data_arrays)}.')
        if t>=self.data_arrays[arr_idx].shape[0]:
            raise ValueError(f'Argument i to TimeSeriesData class method get_causal_Xy_i() was provided as {i} but must be less than {self.data_arrays[arr_idx].shape[0]}.')
        data_i = self.data_arrays[arr_idx]
        X_new = []
        if parents is not None and parents!=[]:
            y_cor = [self.var_name2index(name) for (name,j) in parents] # y coordinate
            x_cor = [t+j for (_,j) in parents] # x coordinate
            xt = data_i[x_cor, y_cor]
            X_new.append(xt)
            X_new = np.stack(X_new)
        else:
            X_new = None
            return None, None

        Y_new = data_i[:,self.var_name2index(target_var)][t]
        return X_new, Y_new.reshape(-1)

