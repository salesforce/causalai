import pandas as pd
import numpy as np
from causalai.data.tabular import TabularData
from causalai.data.transforms.tabular import StandardizeTransform, Heterogeneous2DiscreteTransform
from typing import List, Dict, Union, Optional

def rca_preprocess(
    data: List[pd.DataFrame],
    time_metric: Union[List[int], np.ndarray],
    time_metric_name: str='time'
):
    '''
    Preprocess pandas data for root cause detection.
    :param data: list of dataframes containing before and after metrics
    :type data: List[pd.DataFrame]
        Examples:[before_incident_dataframe, after_incident_dataframe]
    :param time_metric: surrogate of time metric (e.g. time index)
    :type time_metric: Union[List[int], np.ndarray]
    :param time_metric_name: name of the metric that represents time-varying context (e.g. time index)
    :type time_metric_name: str
    :return: TabularData object
    :type: TabularData
    '''
    df = pd.concat(data)
    time_metric = np.array(time_metric)
    df[time_metric_name] = time_metric
    data_array = df.to_numpy()
    var_names = df.columns.tolist()
    transforms = StandardizeTransform()
    transforms.fit(data_array)
    data_transformed = transforms.transform(data_array)
    
    data_obj = TabularData(data_transformed, var_names=var_names)
    
    return data_obj, var_names    

def distshift_detector_preprocess(
    data: List[pd.DataFrame],
    domain_index: Union[List[int], np.ndarray],
    domain_index_name: str='domain_index',
    n_states: int=2
):
    '''
    Preprocess data for causal discovery for heterougenous (discrete/continuous) data.
    :param data: list of dataframes containing variables under different domains
    :type data: List[pd.DataFrame]
    :param domain_index: domain index integers
    :type domain_index: Union[List[int], np.ndarray]
    :param domain_index_name: name of the domain index column
    :type domain_index_name: str
    :param n_states: number of states for discretizing continuous variables
    :type n_states: int
    :return: TabularData object
    :type: TabularData
    '''
    df = pd.concat(data)
    domain_index = np.array(domain_index)
    df[domain_index_name] = domain_index
    data_array = df.to_numpy()
    var_names = df.columns.tolist()
    # Discretize continuous variables
    transforms = Heterogeneous2DiscreteTransform(nstates=n_states)
    discrete = set_discrete_variable(var_names, domain_index_name)
    transforms.fit(data_array, var_names=var_names, discrete=discrete)
    data_transformed = transforms.transform(data_array)
    
    data_obj = TabularData(data_transformed, var_names=var_names)
    
    return data_obj, var_names
    
def set_discrete_variable(
    var_names: List[str], 
    discrete_variable_name: str
):
    '''
    Set discrete variables in the SEM.
    :param var_names: list of variable names
    :type var_names: List[str]
    :param discrete_variable_name: name of the discrete variable
    :type discrete_variable_name: str
    :return: dictionary of discrete variables
    :type: Dict[str, bool]
    '''
    discrete = {name: False for name in var_names}
    discrete[discrete_variable_name] = True
    return discrete