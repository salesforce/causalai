
from __future__ import print_function
from typing import TypedDict, Tuple, List, Union, Optional, Dict
import warnings
from collections import defaultdict
from copy import deepcopy
import numpy as np
import scipy.stats
import time
import math
from numpy import ndarray
from ...data.time_series import TimeSeriesData
try:
    import ray
except:
    pass


class TreatmentInfo(TypedDict):
    var_name: Union[str,int]
    treatment_value: Union[ndarray,int,float]
    control_value: Union[ndarray,int,float]

class ConditionalInfo(TypedDict):
    var_name: Union[str,int]
    condition_value: Union[int,float]




class CausalInference:
    '''
    This class implements causal inference for time series data, for both continuous and discrete data. 
    Specifically, it supports average treatment effect (ATE) and conditional ATE. To perform causal inference,
    this class requires the observational data, causal graph for the data, a prediction model of choice which
    is used for learning the mapping between variables in the causal graph, and specifying whether the data
    is discrete or continuous. This class also supports the use of multi-processing to speed up computation.
    Typically multi-processing is only helpful when the size of the relevant graph (depending on the
    treatment variables and the target variables) is large (mode than 10) or when the prediction model is
    heavy (e.g. MLP).

    '''
    def __init__(self, data: ndarray, var_names: List[Union[str, int]],\
                    causal_graph: Dict[Union[int,str], Tuple[Union[int, str],int]],\
                    prediction_model=None, use_multiprocessing: bool=False, discrete: bool=False):
        '''
        :param data: The observational data of size (N,D) where N is the number of observations and D is the 
            number of variables.
        :type data: ndarray
        :param var_names: List of variable names. The number of variables must be the same as the number of columns
            in data.
        :type var_names: list
        :param causal_graph: The underlyig causal graph for the given data array. causal_graph is a
            dictionary with variable names as keys and the list of parent nodes of each key as
            the corresponding values.
        :type causal_graph: dict
        :param prediction_model: A model class (e.g. Sklearn`s LinearRegression) that has fit and predict method. Do not pass
            an instantiated class object, rather an uninstantiated one. None may be specified when discrete=True, 
            in which case our default prediction model for discrete data is used. Otherwise, For data with linear 
            dependence between variables, typically Sklearn`s LinearRegression works, and for non-linear dependence, 
            Sklearn`s MLPRegressor works.
        :type prediction_model: model class
        :param use_multiprocessing: If true multi-processing is used to speed up computation.
        :type use_multiprocessing: bool
        :param discrete: Set to true if the data array (i.e. all the variables) is discrete. Note that the states
            for a variable must take value in [0,1,...K-1], where K is the number of states for that variable.
            Each variable can have a different number of states.
        :type discrete: bool
        '''

        self.discrete = discrete
        self.data = TimeSeriesData(data, var_names=var_names)
        self.data_processor = _DataProcessor(data, var_names, discrete=discrete)
        self.causal_graph = causal_graph

        assert prediction_model is not None or (prediction_model is None and discrete is True), 'prediction_model must be specified if data is not discrete'
        # when discrete is True and prediction_model is None, use our simple model class DiscretePredictionModel
        self.prediction_model = prediction_model if prediction_model is not None else DiscretePredictionModel
        self.use_multiprocessing = use_multiprocessing

        self.graph_obj = _TimeSeriesGraph(causal_graph, var_names)
        self.node_seq = self.graph_obj.topologicalSort_causal_paths()

        self.train_model = _train_model

    def start(self):
        if self.use_multiprocessing==True and 'ray' in globals():
            if not ray.is_initialized():
                ray.init()
    def stop(self):
        if self.use_multiprocessing==True and 'ray' in globals():
            if ray.is_initialized():
                ray.shutdown()

    def ate(self, target_var: Union[int,str],\
                  treatments: Union[TreatmentInfo, List[TreatmentInfo]]) -> Tuple[float, ndarray, ndarray]:
        '''
        :param target_var: Specify the name of the target variable of interest on which the effect of the treatment is to be estimated.
        :type target_var: int or str
        :param treatments: Each treatment is specified as a dictionary in which the keys are var_name, treatment_value, control_value.
            The value of var_name is a str or int depending on var_names specified during class object creation, and
            treatment_value and control_value are 1D arrays of length equal to the number of observations in data (specified
            during class object creation).
        :type treatments: dict or list of dict

        :return: Returns a tuple of 3 items:

            - ate: The average treatment effect on target_var.

            - y_treat: The individual effect of treatment value for each observation.

            - y_treat: The individual effect of control value for each observation.
        :rtype: float, ndarray, ndarray
        '''

        if type(treatments)!=list:
            treatments = [treatments]
        for treatment_i in treatments:
            if type(treatment_i['treatment_value']) in [int, float]:
                treatment_i['treatment_value'] = treatment_i['treatment_value']* np.ones((self.data.data_arrays[0].shape[0],))
            else:
                assert len(treatment_i['treatment_value'].shape)==1 and len(treatment_i['treatment_value'])==self.data.data_arrays[0].shape[0],\
                    f"treatment_value must be a scalar or 1D array of same length as the data array along index 0. "\
                    f"But found {treatment_i['treatment_value'].shape} and {self.data.data_arrays[0].shape[0]}."
            if type(treatment_i['control_value']) in [int, float]:
                treatment_i['control_value'] = treatment_i['control_value']* np.ones((self.data.data_arrays[0].shape[0],))
            else:
                assert len(treatment_i['control_value'].shape)==1 and len(treatment_i['control_value'])==self.data.data_arrays[0].shape[0],\
                    f"control_value must be a scalar or 1D array of same length as the data array along index 0. "\
                    f"But found {treatment_i['control_value'].shape} and {self.data.data_arrays[0].shape[0]}."

            if self.discrete:
                treatment_var = self.data.index2var_name(treatment_i['var_name'])
                tval = treatment_i['treatment_value']
                cval = treatment_i['control_value']
                num_states = self.data_processor.num_states[self.data.var_names.index(treatment_var)]
                assert np.all(tval<num_states),\
                            f'Treatment variable value must be in range [0,...,{num_states-1}], but found {tval.max()}'
                assert np.all(cval<num_states),\
                            f'Treatment variable value must be in range [0,...,{num_states-1}], but found {cval.max()}'


        target_var = self.data.index2var_name(target_var)
        treatment_vars = [treatment_i['var_name'] for treatment_i in treatments]

        opt_lag = 0
        for child,parents in self.causal_graph.items():
            lag = min(p[1] for p in parents) if len(parents)>0 else 0
            opt_lag = min(opt_lag, lag)
        opt_lag = -opt_lag

        self.models = {}

        relevant_nodes = self.graph_obj.relevant_nodes([target_var])
        relevant_node_seq = [node for node in self.node_seq if node in relevant_nodes]
        relevant_node_seq = relevant_node_seq + [target_var] if relevant_node_seq[-1]!=target_var else relevant_node_seq

        isTreatment_relevant = False
        for t in treatments:
            isTreatment_relevant = isTreatment_relevant or t['var_name'] in relevant_node_seq
        if not isTreatment_relevant:
            print(f"None of the treatment variables {[t['var_name'] for t in treatments]} are not causally affecting the target variable {target_var}.")
            return 0, None, None


        self.start()
        if self.use_multiprocessing==True and 'ray' in globals():
            self.train_model = ray.remote(_train_model) # Ray wrapper; avoiding Ray Actors because they are slower

        for node in relevant_node_seq: # self.node_seq:
            if  node not in treatment_vars and self.causal_graph[node]!=[]:
                if self.use_multiprocessing==True and 'ray' in globals():
                    self.models[node] = self.train_model.remote(self.data, self.data_processor, self.prediction_model, node, self.causal_graph[node]) #model
                else:
                    self.models[node] = self.train_model(self.data, self.data_processor, self.prediction_model, node, self.causal_graph[node]) #model

        if self.use_multiprocessing==True and 'ray' in globals():
            for node in relevant_node_seq: # self.data.var_names:
                if  node not in treatment_vars and self.causal_graph[node]!=[]:
                    self.models[node] = ray.get(self.models[node])

        self.stop()

        treatment_data = deepcopy(self.data)
        control_data = deepcopy(self.data)

        for treatment_i in treatments:
            idx = self.data.var_name2index(treatment_i['var_name'])
            treatment_data.data_arrays[0][:, idx] = treatment_i['treatment_value']
            control_data.data_arrays[0][:, idx] = treatment_i['control_value']

        y_treatment, y_control = [], []
        for j in range(opt_lag, self.data.data_arrays[0].shape[0]):
            for var in relevant_node_seq: # self.data.var_names:
                if var not in treatment_vars and self.causal_graph[var]!=[]:
                    x,_ = treatment_data.get_causal_Xy_i(j, 0, var, self.causal_graph[var]) # x has shape 1xnum_parents
                    if x is not None: # x is None when node has no parents
                        parents = [p[0] for p in self.causal_graph[var]]
                        x = self.data_processor.transformX(x, parents)
                        pred = self.models[var].predict(x)
                        pred = self.data_processor.inv_transform(pred, [var])
                        treatment_data.data_arrays[0][j,self.data.var_name2index(var)] = pred

                        x,_ = control_data.get_causal_Xy_i(j, 0, var, self.causal_graph[var])
                        x = self.data_processor.transformX(x, parents)
                        pred = self.models[var].predict(x)
                        pred = self.data_processor.inv_transform(pred, [var])
                        control_data.data_arrays[0][j,self.data.var_name2index(var)] = pred

            y_treatment.append(treatment_data.data_arrays[0][j, self.data.var_name2index(target_var)])
            y_control.append(control_data.data_arrays[0][j, self.data.var_name2index(target_var)])

        # Compute ATE
        y_treatment = np.array(y_treatment)
        y_control = np.array(y_control)

        # y_treatment = self.data_processor.inv_transform(y_treatment, [target_var])
        # y_control = self.data_processor.inv_transform(y_control, [target_var])

        if len(y_treatment)>0 and len(y_control)>0:
            return (y_treatment.mean() - y_control.mean()), y_treatment, y_control
        print('Not enough samples to perform causal inference.')
        return math.nan, None, None

    def cate(self, target_var: Union[int,str],\
                  treatments: Union[TreatmentInfo, List[TreatmentInfo]],\
                  conditions: Union[Tuple[ConditionalInfo], ConditionalInfo], condition_prediction_model=None) -> float:
        '''
        :param target_var: Specify the name of the target variable of interest on which the effect of the treatment is to be estimated.
        :type target_var: int or str
        :param treatments: Each treatment is specified as a dictionary in which the keys are var_name, treatment_value, control_value.
            The value of var_name is a str or int depending on var_names specified during class object creation, and
            treatment_value and control_value are 1D arrays of length equal to the number of observations in data (specified
            during class object creation).
        :type treatments: dict or list of dict
        :param conditions: Each condition is specified as a dictionary in which the keys are var_name, and condition_value.
            The value of var_name is a str or int depending on var_names specified during class object creation, and
            condition_value is a scalar value (float for continuous data and integer for discrete data).
        :type conditions: dict or list of dict
        :param condition_prediction_model: A model class (e.g. Sklearn`s LinearRegression) that has fit and predict method. Do not pass
            an instantiated class object, rather an uninstantiated one. None may be specified when discrete=True, 
            in which case our default prediction model for discrete data is used. Otherwise, For data with linear 
            dependence between variables, typically Sklearn`s LinearRegression works, and for non-linear dependence, 
            Sklearn`s MLPRegressor works.
        :type condition_prediction_model: model class

        :return: Returns CATE-- The conditional average treatment effect on target_var.
        :rtype: float
        '''
        target_var = self.data.index2var_name(target_var)
        relevant_nodes = self.graph_obj.relevant_nodes([target_var])
        if type(conditions)!=list:
            conditions = [conditions]

        if self.discrete:
            for conditions_i in conditions:
                var_idx = self.data.var_name2index(conditions_i['var_name'])
                c = conditions_i['condition_value']
                num_states = self.data_processor.num_states[var_idx]
                assert np.all(c<num_states),\
                                f'Condition variable value must be in range [0,...,{num_states-1}], but found {c}'


        ate, y_treat, y_control = self.ate(target_var, treatments)
        if y_treat is None:
            return math.nan

        if self.discrete and condition_prediction_model is not None:
            idx = set()
            for i, conditions_i in enumerate(conditions):
                var_idx = self.data.var_name2index(conditions_i['var_name'])
                data_c_i = self.data.data_arrays[0][:, var_idx][-len(y_treat):]
                c = conditions_i['condition_value']
                idx_i = set(np.where(data_c_i==c)[0])
                idx = idx & idx_i if i>0 else idx_i # intersection
            idx = list(idx)
            if len(idx)==0:
                print('No observational data was found that satisfied the provided conditions.')
                return math.nan
            cate = (y_treat[idx] - y_control[idx]).mean()
            return cate

        # if data is not discrete, follow the procedure below
        assert condition_prediction_model is not None, 'condition_prediction_model must not be None for non-discrete data'
        y_treat = self.data_processor.transformY(y_treat, target_var)
        y_control = self.data_processor.transformY(y_control, target_var)

        condition_obs = []
        condition_given = []
        for conditions_i in conditions:
            var_idx = self.data.var_name2index(conditions_i['var_name'])
            var_name = self.data.index2var_name(conditions_i['var_name'])
            v = self.data.data_arrays[0][:, var_idx]
            v = self.data_processor.transformX(v.reshape(-1,1), [var_name])
            condition_obs.append(v)

            v = np.array(conditions_i['condition_value']).reshape(1,1)
            v = self.data_processor.transformX(v, [var_name])
            condition_given.append(v)

        condition_obs = np.hstack(condition_obs)
        condition_given = np.hstack(condition_given)

        model_treat = deepcopy(condition_prediction_model)()
        model_treat.fit(condition_obs[-y_treat.shape[0]:], y_treat)

        model_control = deepcopy(condition_prediction_model)()
        model_control.fit(condition_obs[-y_control.shape[0]:], y_control)

        t = model_treat.predict(condition_given)
        c = model_control.predict(condition_given)
        cate = (t - c)[0]
        cate = self.data_processor.inv_transform(cate, [target_var])
        return cate


############# Helper functions/classes below #############

class DiscretePredictionModel:
    def __init__(self):
        pass
    def fit(self, X, y):
        self.num_states = num_states = [len(np.unique(X[:,i])) for i in range(X.shape[1])]
        p_y_x = np.zeros((len(np.unique(y)), sum(num_states)))
        for i,yi in enumerate(np.unique(y)):
            idx = np.where(y==yi)[0]
            for x_idx in range(X.shape[1]):
                x_=X[idx, x_idx]
                start_idx = sum(num_states[:x_idx])
                for j,x_idx_j in enumerate(range(self.num_states[x_idx])):
                    p_yi_xj = len(np.where(x_==x_idx_j)[0])/len(x_)
                    p_y_x[i,start_idx+j] = p_yi_xj
        self.prob_table = p_y_x
    def predict(self, X):
        p = 0
        y = None
        for i in range(self.prob_table.shape[0]):
            p_i = 1.
            for x_idx in range(X.shape[1]):
                start_idx = sum(self.num_states[:x_idx])
                p_i *= self.prob_table[i, start_idx+X[0, x_idx]]
            if p_i>=p:
                p = p_i
                y = i
        return y

def _train_model(data, data_processor, prediction_model, node, parents):
    X,y, column_names = data.get_causal_Xy(target_var=node, parents=parents) # X is a 2D array, y is a 1D array
    parents = [p[0] for p in parents]
    if X is not None: # X is None when node has no parents
        X = data_processor.transformX(X, parents)
        y = data_processor.transformY(y, node)
        model = deepcopy(prediction_model)()
        m=model.fit(X,y)
        return model
    return None

class _DataProcessor:
    def __init__(self, data, var_names, discrete=False):
        self.discrete = discrete
        self.var_names = var_names
        if discrete:
            data_ = np.array(data, dtype=int)
            assert np.all(data==data_), 'The provided data array must have integer entries discrete is set to True'
            data = data_
            self.num_states = np.array(np.max(data, axis=0), dtype=int)+1
            data = self.discretize_all(data, var_names)

        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True) + 1e-7

        
    def discretize(self, x, var_name):
        '''
        x is a 1D vector
        '''
        nstates = self.num_states[self.var_names.index(var_name)]
        out = np.zeros((x.shape[0], nstates))
        out[range(x.shape[0]), x] = 1
        return out
    def discretize_all(self, x, var_names):
        idx = [self.var_names.index(name) for name in var_names]
        num_states = self.num_states[idx]
        out = np.zeros((x.shape[0], num_states.sum()))
        for i,n in enumerate(num_states):
            out[:, num_states[:i].sum():num_states[:i].sum()+n] = self.discretize(x[:,i], var_names[i])
        return out

    def transformX(self, x, var_names):
        if self.discrete:
            return x    

        idx = [self.var_names.index(name) for name in var_names]
        if len(idx)==1:
            idx = idx[0]
        out = (x - self.mean[0, idx])/self.std[0, idx]
        return out

    def transformY(self, x, var_name):
        if self.discrete:
            out = x
        else:
            idx = self.var_names.index(var_name)
            out = (x - self.mean[0, idx])/self.std[0, idx]
        return out

    def inv_transform(self, x, var_names):
        out = x
        if not self.discrete:
            idx = [self.var_names.index(name) for name in var_names]
            if len(idx)==1:
                idx = idx[0]
            out = x* self.std[0, idx] + self.mean[0, idx]
        return out

class _TimeSeriesGraph:
    def __init__(self, G, var_names):
        self.causal_graph = G
        self.var_names = var_names
        self.num_nodes = len(var_names)
        self.construct_full_graph_dict()
        
        
        graph = self.get_nonlagged_graph()
        self.adj_graph_nonlagged, _ = self.get_adjacency_graph(graph)
        assert self.isCyclic() is False,\
                'The given causal_graph has a cycle among non-lagged connections! Such cycles are not allowed.'

    def construct_full_graph_dict(self):
        '''
        Verify that all nodes in causal_graph are listed in var_names, and if 
        any node is missing in causal_graph.keys(), add it with an empty list 
        of parents as the corresponding value.
        '''
        all_nodes = []
        for child, parents in self.causal_graph.items():
            if child not in all_nodes:
                all_nodes.append(child)

            for parent in parents:
                if parent not in all_nodes:
                    all_nodes.append(parent[0])
        all_nodes = set(all_nodes)
        assert len(all_nodes) - len(set(self.var_names))==0,\
            f'Oops, there are nodes in the causal_graph ({(all_nodes) - (set(self.var_names))}) which are '\
            f'missing in var_names! var_names must contain all the nodes.'
                    
        for node in self.var_names:
            if node not in self.causal_graph.keys():
                self.causal_graph[node] = []

    def _isCyclic(self, v, visited, visited_during_recusion):
        visited[v] = True
        visited_during_recusion[v] = True

        for child in self.adj_graph_nonlagged[self.var_names[v]]:
            child = self.var_names.index(child)
            if visited[child] == False:
                if self._isCyclic(child, visited, visited_during_recusion) == True:
                    return True
            elif visited_during_recusion[child] == True:
                return True

        visited_during_recusion[v] = False
        return False

    def isCyclic(self):
        '''
        Check if the causal graph has cycles among the non-lagged connections
        '''
        visited = [False] * (self.num_nodes + 1)
        visited_during_recusion = [False] * (self.num_nodes + 1)
        for node in range(self.num_nodes):
            if visited[node] == False:
                if self._isCyclic(node,visited,visited_during_recusion) == True:
                    return True
        return False

    def relevant_nodes(self, targets):
        '''
        Given a target node, return all the ansestors of this node in the causal_graph
        targets: list
        '''
        l = []
        q = targets
        seen = set()
        while q!=[]:
            node = q.pop()
#             seen.add(node)
            parents = self.causal_graph[node]
            for p in parents:
                if p[0] not in seen:
                    l.append(p[0])
                    q.append(p[0])
                    seen.add(p[0])
        return l

    def get_nonlagged_graph(self):
        '''
        Return only non-lagging subset of parents in the graph for each node
        '''
        g = {}
        for n in self.var_names:
            g[n] = []
        for child, parents in self.causal_graph.items():
            for p in parents:
                if p[1]==0:
                    g[child].append(p[0])
        return g
    
    def get_adjacency_graph(self, graph):
        '''
        Given graph where keys are children and values are parents, convert to adjacency graph where
        keys are parents and values are children.
        '''
        ad_graph = dict()
        all_nodes = []
        for child, parents in graph.items():
            if child not in all_nodes:
                all_nodes.append(child)

            for parent in parents:
                if parent not in all_nodes:
                    all_nodes.append(parent)

                if parent in ad_graph:
                    ad_graph[parent].append(child)
                else:
                    ad_graph[parent] = [child]
        for node in all_nodes:
            if node not in ad_graph.keys():
                ad_graph[node] = []
        return ad_graph, all_nodes

    def topologicalSort_causal_paths(self):
        '''
        Given a causal graph, list of all the nodes in the graph, return the topologically sorted list of nodes.
        '''
        def sortUtil(graph, n,visited,stack, all_nodes):
            visited[all_nodes.index(n)] = True
            for element in graph[n]:
                if visited[all_nodes.index(element)] == False:
                    sortUtil(graph, element,visited,stack, all_nodes)
            stack.insert(0,n)
        
        visited = [False]*len(self.var_names)
        stack =[]
        for element in self.var_names:
            if visited[self.var_names.index(element)] == False:
                sortUtil(self.adj_graph_nonlagged, element, visited,stack, self.var_names)
        valid_node_seq = []
        for e in stack:
            valid_node_seq.append(e)
        return valid_node_seq

