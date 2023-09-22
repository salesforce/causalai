
from abc import abstractmethod
from typing import TypedDict, Tuple, List, Union, Optional, Dict
from numpy import ndarray
import numpy as np
from ...data.tabular import TabularData
from ...models.common.prior_knowledge import PriorKnowledge
import warnings
try:
    import ray
except:
    pass

class ResultInfoTabularSingle(TypedDict):
    parents: List[Union[str,int]]
    value_dict: Dict[Union[str,int], float]
    pvalue_dict: Dict[Union[str,int], float]
    undirected_edges: List[Union[str,int]]

class ResultInfoTabularMB(TypedDict):
    mb: List[Union[str,int]]
    value_dict: Dict[Union[str,int], float]
    pvalue_dict: Dict[Union[str,int], float]
    undirected_edges: List[Union[str,int]]

class ResultInfoTabularFull(TypedDict):
    parents: List[Union[str,int]]
    value_dict: Dict[Union[str,int], float]
    pvalue_dict: Dict[Union[str,int], float]

class BaseTabularAlgo(object):
    def __init__(self, data: TabularData, prior_knowledge: Optional[PriorKnowledge]=None, **kargs):
        '''
        :param data: It contains data.values, a numpy array of shape (observations N, variables D).
        :type data: TabularData object 

        :param prior_knowledge: Specify prior knoweledge to the causal discovery process by either
            forbidding links/co-parents that are known to not exist, or adding back links/co-parents that do exist
            based on expert knowledge. See the PriorKnowledge class for more details.
        :type prior_knowledge: PriorKnowledge object
        '''
        self.data = data
        prior_knowledge = prior_knowledge if prior_knowledge is not None else PriorKnowledge()
        self.prior_knowledge = prior_knowledge
        self.__dict__.update(kargs)

        # Initialize the dictionaries for the pvalue_dict, value_dict
        self.pvalue_dict = dict()
        self.value_dict = dict()

    def start(self, full_cd: bool=False):
        if self.use_multiprocessing==True and 'ray' in globals() and not full_cd:
            if not ray.is_initialized():
                ray.init()
    def stop(self, full_cd: bool=False):
        if self.use_multiprocessing==True and 'ray' in globals() and not full_cd:
            if ray.is_initialized():
                ray.shutdown()

    def get_all_parents(self, target_var: Union[int, str]) -> List:
        '''
        Populates the list using all nodes
        '''
        candidate_parents = [n for n in self.data.var_names if n!=target_var]
        return candidate_parents

    def get_candidate_parents(self, target_var: Union[int, str]) -> List:
        '''
        Populates the list using all the nodes that prior_knowledge allows
        '''
        candidate_parents = [n for n in self.data.var_names if self.prior_knowledge.isValid(parent=n, child=target_var) and n!=target_var]
        return candidate_parents

    def get_candidate_mb(self, target_var: Union[int, str]) -> List:
        '''
        Populates the list using all the nodes that prior_knowledge allows
        '''
        candidate_parents = self.get_candidate_parents(target_var)
        candidate_children = [n for n in self.data.var_names if self.prior_knowledge.isValid(parent=target_var, child=n) and n!=target_var]
        candidate_co_parents = [n for n in self.data.var_names if n!=target_var and self.prior_knowledge.isValid_co_parent(first_co_parent=target_var, second_co_parent=n)]
        return list(set(candidate_parents+candidate_children+candidate_co_parents))

    def sort_parents(self, parents_vals: Dict) -> Tuple[Union[int, str]]:
        """
        Sort (in descending order) parents according to test statistic values.

        :param parents_vals: Dictionary of form {<var_name>:float, ...} containing the test
            statistic value of each causal link.
        :type parents_vals: dict
        :return: List of form [<var_i_name>, <var_k_name>, ...] containing sorted parents.
        :rtype: list
        """
        # Get the absolute value for all the test statistics
        abs_values = {k: np.abs(parents_vals[k]) for k in list(parents_vals)}
        return sorted(abs_values, key=abs_values.get, reverse=True)

    def get_parents(self, pvalue_thres: float= 0.05) -> Tuple[Union[int, str]]:
        '''
        Assuming run() function has been called for a target_var, get_parents function returns the list of 
        parent names that cause the target_var under the given pvalue_thres.

        :param pvalue_thres: Significance level used for hypothesis testing (default: 0.05). 
            Candidate parents with pvalues above pvalue_thres are ignored, and the rest are returned as the 
            cause of the target_var.
        :type pvalue_thres: float

        :return: List of estimated parents.
        :rtype: list
        '''
        parents_values = dict()
        nonsignificant_parents = list()

        for index_parent, parent in enumerate(self.pvalue_dict.keys()):

            parents_values[parent] = np.abs(self.value_dict[parent]) if self.value_dict[parent] is not None else np.abs(self.pvalue_dict[parent])

            if self.pvalue_dict[parent] > pvalue_thres:
                nonsignificant_parents.append(parent)

        # Remove non-significant links
        for parent in nonsignificant_parents:
            del parents_values[parent]
        parents = self.sort_parents(parents_values)
        return parents

    @abstractmethod
    def run(self, target_var: Union[int,str], pvalue_thres: float=0.05) -> Union[ResultInfoTabularSingle,ResultInfoTabularMB]:
        '''
        Run causal discovery using the algorithm implemented here

        :param target_var: Target variable index or name for which parents need to be estimated.
        :type target_var: int
        :param pvalue_thres: Significance level used for hypothesis testing (default: 0.05). 
            Candidate parents with pvalues above pvalue_thres are ignored, and the rest are returned as the 
            cause of the target_var.
        :type pvalue_thres: float

        :return: Dictionary has three keys:

            - parents or markov_blanket : List of estimated parents or markov blanket.

            - value_dict : Dictionary of form {var3_name:float, ...} containing the test statistic of a link.

            - pvalue_dict : Dictionary of form {var3_name:float, ...} containing the p-value corresponding to the above test statistic.
        :rtype: dict
        '''
        raise NotImplementedError()

class BaseTabularAlgoFull(object):
    def __init__(self, **kargs):
        self.__dict__.update(kargs)
        self.result = {}

    def start(self):
        if self.use_multiprocessing==True and 'ray' in globals():
            if not ray.is_initialized():
                ray.init()
    def stop(self):
        if self.use_multiprocessing==True and 'ray' in globals():
            if ray.is_initialized():
                ray.shutdown()
                
    def get_parents(self, pvalue_thres: float=0.05, 
                    target_var: Optional[Union[int,str]]=None) -> Dict[Union[int,str], Tuple[Union[int,str]]]:
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

        :return: Dictionary has D keys, where D is the number of variables. The value corresponding each key is
            the list of lagged parent names that cause the target variable under the given pvalue_thres.
        :rtype: dict
        '''
        all_parents = {}

        for target_var_name in self.result.keys():
            parents_values = dict()
            nonsignificant_parents = list()

            for index_parent, parent in enumerate(self.result[target_var_name]['pvalue_dict'].keys()):

                parents_values[parent] = np.abs(self.result[target_var_name]['value_dict'][parent])

                if self.result[target_var_name]['pvalue_dict'][parent] > pvalue_thres:
                    nonsignificant_parents.append(parent)

            # Remove non-significant links
            for parent in nonsignificant_parents:
                del parents_values[parent]
            parents = self.sort_parents(parents_values)
            all_parents[target_var_name] = parents

        if target_var is None:
            return all_parents
        assert target_var in self.data.var_names, f'{target_var} not found in the variable names specified for the data!'
        return all_parents[target_var]

    def run(self, pvalue_thres: float=0.05) -> Dict[Union[int,str],ResultInfoTabularFull]:
        """
        Run causal discovery using the algorithm implemented here for estimating the causal stength of all 
        potential parents of all the variables.

        :param pvalue_thres: Significance level used for hypothesis testing (default: 0.05). Candidate parents with pvalues above pvalue_thres
            are ignored, and the rest are returned as the cause of the target_var.
        :type pvalue_thres: float

        :return: Dictionay has D keys, where D is the number of variables. The value corresponding each key is 
            the dictionary output of BaseTabularAlgo.run.
        :rtype: dict
        """
        raise NotImplementedError()
