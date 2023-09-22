import warnings
import numpy as np
import copy
from typing import Callable
from sklearn.preprocessing import KBinsDiscretizer

def GenerateRandomTimeseriesSEM(var_names=['a', 'b', 'c', 'd', 'e'], max_num_parents=4, max_lag=4, seed=0, fn:Callable = lambda x:x, coef: float = 0.1):
    '''
    Generate a random structural equation model (SEM) for time series data.

    :param var_names: Names of variables in the SEM in the form of a list of str.
    :type var_names: list
    :param max_num_parents: Maximum number of causal parents allowed in the randomly generated SEM.
    :type max_num_parents: int
    :param max_lag: Maximum time lag between parent and child variable allowed in the randomly generated SEM. Must be non-negative.
    :type max_lag: int
    :param seed: Random seed used for reproducibility.
    :type seed: int
    :param fn: Function applied to a parent variable when generating child variable data. Default: Linear 
        function for linear causal relation.
    :type fn: Callable
    :param coef: coefficient of parent variables in the randomly generated SEM.
    :type coef: float
    '''
    np.random.seed(seed)
    # coef = 0.1 # larger values may cause exploding values in data array for some seeds
    sem_dict = {n:[] for n in var_names}
    max_num_parents = min(max_num_parents, len(var_names))
    for n in var_names:
        num_parents = np.random.randint(1, max_num_parents+1)
        parents = np.random.permutation(var_names)[:num_parents]
        lags = np.random.randint(-max_lag, 0, num_parents)
        sem_dict[n] = [((p,int(l)),coef, fn) for p,l in zip(parents, lags)]
    return sem_dict

def GenerateRandomTabularSEM(var_names=['a', 'b', 'c', 'd', 'e', 'f'], max_num_parents=4, seed=0, fn:Callable = lambda x:x, coef: float = 0.1):
    '''
    Generate a random structural equation model (SEM) for tabular data using the following procedure:
    Randomly divide variables into non-overlapping groups of size between 3 and num_vars. Then randomly
    create edges between a preceeding group and a following group such that max_num_parents is never exceeded.

    :param var_names: Names of variables in the SEM in the form of a list of str.
    :type var_names: list
    :param max_num_parents: Maximum number of causal parents allowed in the randomly generated SEM.
    :type max_num_parents: int
    :param seed: Random seed used for reproducibility.
    :type seed: int
    :param fn: Function applied to a parent variable when generating child variable data. Default: Linear 
        function for linear causal relation.
    :type fn: Callable
    :param coef: coefficient of parent variables in the randomly generated SEM.
    :type coef: float
    '''
    np.random.seed(seed)
    num_vars = len(var_names)
    groups_nums = []
    while num_vars>0:
        n = np.random.randint(1, min(3,num_vars+1))
        num_vars -= n
        groups_nums.append(n)
    vars_unused = copy.copy(var_names)
    num_levels = len(groups_nums)
    groups = []
    for i in range(num_levels):
        idx = np.random.permutation(len(vars_unused))[:groups_nums[i]]
        group = [vars_unused[i] for i in idx]
        groups.append(group)
        for v in group:
            del vars_unused[vars_unused.index(v)]
    # coef = 0.1 # larger values may cause exploding values in data array for some seeds
    
    sem_dict = {n:[] for n in var_names}
    for i, children in enumerate(groups[1:]):
        parents = groups[i]
        for c in children:
            sem_dict[c] = [(p,coef, fn) for p in parents[:np.random.randint(1, max(2, min(max_num_parents,len(parents)) ))]]
        
        add_ancestor_as_parent = np.random.rand()>0.2
        if add_ancestor_as_parent:
            r = np.random.randint(0,max(1, i))
            if r!=i:
                parents = groups[r][:np.random.randint(1, max(2, min(max_num_parents,len(parents))))]
                sem_dict[c].extend([(p,coef, fn) for p in parents])
    for n in var_names:
        if n not in list(sem_dict.keys()):
            sem_dict[n] = []
    return sem_dict

def GenerateSparseTimeSeriesSEM(var_names=['a', 'b', 'c', 'd', 'e'], graph_density=0.1, max_lag=4, seed=0, fn:Callable = lambda x:x, coef: float = 0.1):
    '''
    Generate a structural equation model (SEM) for time series data using the following procedure:
    For N nodes, enumerate them from 0-N. For each time lag (until max_lag), for all i,j between 0-N, if i < j, 
    the edge from vi to vj exists with probability graph_density, and if i >= j there cannot be an edge 
    betwen them.

    :param var_names: Names of variables in the SEM in the form of a list of str.
    :type var_names: list
    :param graph_density: Probability that an edge between node i and j exists.
    :type graph_density: float in the range (0,1]
    :param max_lag: Maximum time lag between parent and child variable allowed in the randomly generated SEM.
    :type max_lag: int
    :param seed: Random seed used for reproducibility.
    :type seed: int
    :param fn: Function applied to a parent variable when generating child variable data. Default: Linear 
        function for linear causal relation.
    :param coef: coefficient of parent variables in the randomly generated SEM. Note: larger values may 
        cause exploding values in data array for some seeds.
    :type fn: Callable
    '''
    np.random.seed(seed)
    num_vars = len(var_names)
    
    sem_dict = {n:[] for n in var_names}
    for lag in range(1, max_lag):
        
        M = np.random.binomial(1, graph_density, (num_vars, num_vars))
        mask = np.ones_like(M, dtype=float)
        mask[np.tril_indices_from(mask)] = 0.
        M = M* mask
    
        for pi in range(num_vars):
            for ci in range(pi+1, num_vars):
                if M[pi, ci] == 1:
                    sem_dict[var_names[ci]].append(((var_names[pi], -lag),coef, fn))
    for n in var_names:
        if n not in list(sem_dict.keys()):
            sem_dict[n] = []
    return sem_dict

def GenerateSparseTabularSEM(var_names=['a', 'b', 'c', 'd', 'e', 'f'], graph_density=0.1, seed=0, fn:Callable = lambda x:x, coef: float = 0.1):
    '''
    Generate a structural equation model (SEM) for tabular data using the following procedure:
    For N nodes, enumerate them from 0-N. For all i,j between 0-N, if i < j, the edge from vi 
    to vj exists with probability graph_density, and if i >= j there cannot be an edge betwen them.

    :param var_names: Names of variables in the SEM in the form of a list of str.
    :type var_names: list
    :param graph_density: Probability that an edge between node i and j exists.
    :type graph_density: float in the range (0,1]
    :param seed: Random seed used for reproducibility.
    :type seed: int
    :param fn: Function applied to a parent variable when generating child variable data. Default: Linear 
        function for linear causal relation.
    :type fn: Callable
    :param coef: coefficient of parent variables in the randomly generated SEM.
    :type coef: float
    '''
    np.random.seed(seed)
    num_vars = len(var_names)
    M = np.random.binomial(1, graph_density, (num_vars, num_vars))
    mask = np.ones_like(M, dtype=float)
    mask[np.tril_indices_from(mask)] = 0.
    M = M* mask
    
    sem_dict = {n:[] for n in var_names}
    for pi in range(num_vars):
        for ci in range(pi+1, num_vars):
            if M[pi, ci] == 1:
                sem_dict[var_names[ci]].append((var_names[pi],coef, fn))
    for n in var_names:
        if n not in list(sem_dict.keys()):
            sem_dict[n] = []
    return sem_dict

def DataGenerator(sem_dict, T, noise_fn=None, intervention=None, discrete=False, nstates=10, seed=1):
    """
    Use structural equation model to generate time series and tabular data.

    :param sem_dict: Structural equation model (SEM) provided as a dictionary
        of format: {<child node>: (<parent node>, coefficient, function)}. Notice that
        the values are lists containing tuples of the form (<parent node>, coefficient, function).
        The parent node must be a tuple of size 2 for time series data, where the 1st element (str) 
        specifies the parent node, while the 2nd element is the time lag (must be non-positive). If this lag is 0,
        it means the the parent is at the same time step as the child (instantaneous effect). For tabular data,
        the parent node is simply a str. Finally, the coefficient must 
        be of type float, and func is a python callable function that takes one float argument as
        input. For example in sem_dict = {a:[((b, -2), 0.2, func)], b:[((b, -1), 0.5, func),...]}, 
        the item a:[((b,-2), 0.2, func)] implies:
        v(a,t) = func(v(b, -2)*0.2), where the v(a,t) denotes the value of node a at time t, and
        v(b, -2) denotes the value of node b at time t-2. For tabular data example 
        sem_dict = {a:[(b, 0.2, func)], b:[(c, 0.5, func),...]}, 
        the item a:[(b, 0.2, func)] implies: v(a) = func(v(b)*0.2).
    :type sem_dict: dict
    :param T: Number of samples.
    :type T: int
    :param noise_fn: List of functions each of which takes t as input and that returns a random vector of length t.
        (default: list of np.random.randn)
    :type noise_fn: list of callables, optional
    :param intervention: Dictionary of format: {1:np.array, ...} containing only keys of intervened
        variables with the value being the array of length T with interventional values.
        Set values to np.nan to leave specific time points of a variable un-intervened.
    :type intervention: dict
    :param discrete: When bool, it specifies whether all the variables are discrete or all of them are continuous.
        If true, the generated data is discretized into nstates uniform bins (default: False). Alternatively, if discrete is specified 
        as a dictionary, then the keys of this dictionary must be the variable names and the value corresponding to
        each key must be True or False. A value of False implies the variable is continuous, and discrete otherwise.
    :type discrete: bool or dict
    :param nstates:  When discrete is True, the nstates specifies the number of bins to use for discretizing
        the data (default=10).
    :type nstates: int
    :param seed: Set the seed value for random number generation for reproduciblity (default: 1).
    :type seed: int

    :return: A tuple of 3 items--

        - data: Generated data array of shape (T, number of variables).

        - var_names: List of variable names corresponding to the columns of data

        - graph: The causal graph that was used to generate the data array. graph is a dictionary with variable names as keys and the list of parent nodes of each key as the corresponding values.
    :rtype: tuple[ndarray, list, dict]

    """
    np.random.seed(seed)
    var_names = list(sem_dict.keys())
    if type(discrete)==dict:
        assert set(var_names)==set(list(discrete.keys())), f'The keys of the argument discrete must match the keys of the argument sem_dict.'
    elif type(discrete)==bool:
        discrete = {name: discrete==True for name in var_names}
    else:
        raise ValueError(f"The argument discrete must be either of type bool or Python dictionary, but found {type(discrete)}.")
    is_any_discrete = any(discrete.values())
    sem_dict, data_type = _standardize_graph(sem_dict)
    sem_dict = _graph_names2num(sem_dict)
    graph_gt = {var_names[n]:None for n in sem_dict.keys()}
    for n in sem_dict:
        if data_type=='time_series':
            p_list = [(var_names[p[0][0]], p[0][1]) for p in sem_dict[n]]
        else:
            p_list = [var_names[p[0][0]] for p in sem_dict[n]]
        graph_gt[var_names[n]] = p_list

    D = len(sem_dict.keys()) # number of variables
    noise_fn = [np.random.randn for _ in range(D)] if noise_fn is None else noise_fn

    # Sanity check
    if D != len(noise_fn):
        raise ValueError(f"noise_fn must be a list of length {D} but found {len(noise_fn)}.")
    max_lag = 0
    instantaneous_graph = _Graph(D)
    for i in range(D):
        for ((node, lag), coef, func) in sem_dict[i]:
            if lag > 0 or type(lag) != int:
                raise ValueError(f"lag must be a non-positive int but found {lag}.")
            if type(coef)!=float:
                raise ValueError(f"coef must be of type float but found {coef}.")
            max_lag = max(max_lag, abs(lag))

            if node!=i and lag==0:
                instantaneous_graph.addEdge(node, i)

    if instantaneous_graph.isCyclic(): 
        raise ValueError("Graph contains cycles! Only DAGs are allowed.")
        #("Intantaneous edges among nodes with lag 0 specified in sem_dict cannot contain cycles.")

    causal_node_seq = instantaneous_graph.topologicalSort()

    if intervention is not None:
        for name in intervention.keys():
            if str(name) not in var_names:
                raise ValueError(f'Intervention is specified for variable name {name}, but it does not exist in the specified graph with node names {var_names}!')
            if len(intervention[name])!=T:
                raise ValueError(f"intervention data for {name} must be of length T, but found {len(intervention[name])}")
            if discrete[name]==True:
                assert np.all(intervention[name]<nstates),\
                            f'Treatment variable value must be in range [0,...,{nstates-1}], but found {intervention[name].max()}'

    history_len = T//5
    data = np.zeros((T+history_len, D)) # data with intervention, if any; has discrete variabless if specified by the discrete argument
    data_orig = np.zeros((T+history_len, D)) # data without intervention; continuous data

    # start with filling all elements with I.I.D. noise
    for i in range(D):
        data[:, i] = noise_fn[i](T+history_len)
        data_orig[:, i] = copy.copy(data[:, i])

    for t in range(max_lag, T+history_len):
        for i in causal_node_seq:
            for ((node, lag), coef, func) in sem_dict[i]:
                # print(f'data_orig[{t},{var_names[i]}] += {coef}. data_orig[{t+lag},{var_names[node]}]')
                data_orig[t, i] += coef * func(data_orig[t + lag, node])

    if is_any_discrete:
        DiscretizeData_orig = _DiscretizeData()
        _ = DiscretizeData_orig(data_orig, data_orig, nstates, compute_bin_range=True)
        
    for t in range(max_lag, T+history_len):
        for i in causal_node_seq:
            node_name = var_names[i]
            if (intervention is not None and node_name in intervention and t >= history_len
                    and not np.isnan(intervention[node_name][t - history_len])):
                if discrete[node_name]==True:
                    val = DiscretizeData_orig.inv_transform(i, intervention[node_name][t - history_len])
                    data[t, i] = val
                else:
                    data[t, i] = intervention[node_name][t - history_len]
                continue

            for ((node, lag), coef, func) in sem_dict[i]:
                data[t, i] += coef * func(data[t + lag, node])

    

    data = data[history_len:]
    data_orig = data_orig[history_len:]
    if is_any_discrete:
        DiscretizeData_ = _DiscretizeData()
        # data in the input below is data with intervetions; data_orig is the continuous data on which bins are estimated for discretization
        data_all_discrete = DiscretizeData_(data, data_orig, nstates)
        for name in var_names:
            if discrete[name] is True:
                data[:, var_names.index(name)] = data_all_discrete[:, var_names.index(name)]

    return data, var_names, graph_gt


def ConditionalDataGenerator(T, data_type='time_series', noise_fn=None, intervention=None, discrete=False, nstates=10, seed=1):
    """
    Generate data that is useful for testing CATE (conditional average treatment estimation) for causal inference.

    The data is generated using the following structural equation model:

    C = noise

    W = C + noise

    X = C*W + noise

    Y = C*X + noise

    Note that the depence between variables in the generated data is instantaneous only (no time lagged dependence)
    for simplicity. Hence this data can be used both for tabular and timeseries cases.

    :param T: Number of samples.
    :type T: int
    :param data_type: String (time_series, or tabular) that specifies whether the generated data causal graph should be 
        specified as tabular or time series (default: time_series).
    :type data_type: str
    :param noise_fn: List of functions each of which takes t as input and that returns a random vector of length t.
        (default: list of np.random.randn)
    :type noise_fn: list of callables, optional
    :param intervention: Dictionary of format: {``W``:np.array, ...} containing only keys of intervened
        variables with the value being the array of length T with interventional values.
        Set values to np.nan to leave specific time points of a variable un-intervened.
    :type intervention: dict
    :param discrete: When bool, it specifies whether all the variables are discrete or all of them are continuous.
        If true, the generated data is discretized into nstates uniform bins (default: False). Alternatively, if discrete is specified 
        as a dictionary, then the keys of this dictionary must be the variable names and the value corresponding to
        each key must be True or False. A value of False implies the variable is continuous, and discrete otherwise.
    :type discrete: bool or dict
    :param nstates:  When discrete is True, the nstates specifies the number of bins to use for discretizing
        the data (default=10).
    :type nstates: int
    :param seed: Set the seed value for random number generation for reproduciblity (default: 1).
    :type seed: int

    :return: A tuple of 3 items--

        - data: Generated data array of shape (T, 4).

        - var_names: List of variable names corresponding to the columns of data

        - graph: The causal graph that was used to generate the data array. graph is a dictionary with variable names as keys and the list of parent nodes of each key as the corresponding values.
    :rtype: tuple[ndarray, list, dict]

    """
    np.random.seed(seed)
    assert data_type in ['time_series', 'tabular'], f'data_type must be one of [time_series, tabular], but found {data_type}!'
    def generate_data(T, D, noise_fn):
        data = np.zeros((T, D))

        noise = {'C': noise_fn(T), 'W': noise_fn(T), 'X': noise_fn(T), 'Y': noise_fn(T)}
        data[:, 0] = noise['C']
        data[:,1] = data[:,0] + noise['W']
        data[:,2] = data[:, 0]*data[:,1] + noise['X']
        data[:,3] = data[:, 0]*data[:,2] + noise['Y']
        return data, noise


    
    noise_fn = noise_fn if noise_fn is not None else np.random.randn
    D = 4 # number of variables

    causal_node_seq = nodes = ['C', 'W', 'X', 'Y']

    if type(discrete)==dict:
        assert set(nodes)==set(list(discrete.keys())), f'The keys of the argument discrete must match the keys of the argument sem_dict.'
    elif type(discrete)==bool:
        discrete = {name: discrete==True for name in nodes}
    else:
        raise ValueError(f"The argument discrete must be either of type bool or Python dictionary, but found {type(discrete)}.")


    if data_type=='tabular':
        graph_gt = {'C': [],
                    'W': ['C'],
                    'X': ['C', 'W'],
                    'Y': ['C', 'X']}
    else:
        graph_gt = {'C': [],
                    'W': [('C',0)],
                    'X': [('C',0), ('W',0)],
                    'Y': [('C',0), ('X',0)]}

    is_any_discrete = any(discrete.values())
    if intervention is not None:
        for key in intervention.keys():
            if key not in nodes:
                raise ValueError(f"intervention dictionary must have keys in {nodes} but found {key}.")
            if len(intervention[key])!=T:
                raise ValueError(f"intervention data for {key} must be of length T, but found {len(intervention[key])}")
            if discrete[key]:
                assert np.all(intervention[key]<nstates),\
                            f'Treatment variable value must be in range [0,...,{nstates-1}], but found {intervention[key].max()}'

    
    data, noise_dict = generate_data(T, D, noise_fn)
    data_orig = copy.deepcopy(data)

    DiscretizeData_orig=None
    if is_any_discrete:
        DiscretizeData_orig = _DiscretizeData()
        _ = DiscretizeData_orig(data_orig, data_orig, nstates, compute_bin_range=True)
        

    # apply interventions
    def get_intervention_data(discrete, disc_data_obj, var_idx, var_name, intervention):
        if discrete:
            val = np.array([disc_data_obj.inv_transform(var_idx, intervention[var_name][j]) for j in range(len(intervention[var_name]))])
        else:
            val = intervention[var_name]
        return val

    if intervention is not None:
        intervened_data = np.zeros((T, D))
        # C
        if 'C' in intervention.keys():
            intervened_data[:, 0] = get_intervention_data(discrete['C'], DiscretizeData_orig, 0, 'C', intervention)
        else:
            intervened_data[:, 0] = noise_dict['C']
        if 'W' in intervention.keys():
            intervened_data[:, 1] = get_intervention_data(discrete['W'], DiscretizeData_orig, 1, 'W', intervention)
        else:
            intervened_data[:, 1] = intervened_data[:,0] + noise_dict['W']
        if 'X' in intervention.keys():
            intervened_data[:, 2] = get_intervention_data(discrete['X'], DiscretizeData_orig, 2, 'X', intervention)
        else:
            intervened_data[:, 2] = intervened_data[:,0]*intervened_data[:,1] + noise_dict['X']
        if 'Y' in intervention.keys():
            intervened_data[:, 3] = get_intervention_data(discrete['Y'], DiscretizeData_orig, 3, 'Y', intervention)
        else:
            intervened_data[:, 3] = intervened_data[:,0]*intervened_data[:,2] + noise_dict['Y']
        data = intervened_data
    
    if is_any_discrete:
        DiscretizeData_ = _DiscretizeData()
        data_all_discrete = DiscretizeData_(data, data_orig, nstates)
        for name in nodes:
            if discrete[name] is True:
                data[:, nodes.index(name)] = data_all_discrete[:, nodes.index(name)]
        # data = DiscretizeData_(data, data_orig, nstates)
    return data, nodes, graph_gt

class _DiscretizeData:
    '''
    Discretizes each continuous variable data into specified number of states via binning using the __call__method. The data 
    provided to this method is considered as the training data. This class also supports methods transform and inv_transform.
    The former can be used to transform a test data into discrete states, where the mapping from continuous to discrete 
    state is the one learned on the training data. The inv_transform method converts a given discrete state to continuous 
    state for a given variable by uniformly sampling a float number within the range of the bin corresponding
    to the given state.
    '''
    def __init__(self):
        self.called=False

    def __call__(self, data, train_data, nstates=10, compute_bin_range=False):
        '''
        :param data: ndarray
            The data array that is discretized and returned
        :param train_data: ndarray
            The data array that is used to find bins for discretization that are applied on data.
        :param nstates: int (default=10)
            When discrete is True, the nstates specifies the number of bins to use for discretizing
            the data.
        '''
        self.nstates = nstates
        data_disc = copy.deepcopy(data)
        self.encoders = []
        if compute_bin_range:
            self.bin_range = [{i:(None,None) for i in range(nstates)} for j in range(data.shape[1])]

        for i in range(data.shape[1]):
            enc = KBinsDiscretizer(n_bins=nstates, encode="ordinal", random_state=0)
            enc.fit(train_data[:, i].reshape(-1,1))
            self.encoders.append(enc)
            data_disc[:,i] = enc.transform(data[:, i].reshape(-1,1)).reshape(-1)

            if compute_bin_range:
                for b in range(nstates):
                    idx = np.where(data_disc[:,i]==b)[0]
                    min_, max_  = data[:,i][idx].min(), data[:,i][idx].max()
                    self.bin_range[i][b] = (min_, max_)
        data_disc = np.array(data_disc, dtype='int')
        self.called = True
        return data_disc
    def transform(self, val, idx):
        # val must be an ndarray of shape (N, 1) where N is any integer
        assert self.called, 'DiscretizeData class object must be called with data array before running the transform method.'

        return self.encoders[idx].transform(val)
    def inv_transform(self, idx, state):
        assert state<self.nstates, f'state of variable in method inv_transform must be < {self.nstates} but {state} was provided!'
        assert idx<len(self.encoders), f'idx in method inv_transform must be < {len(self.encoders)} but {idx} was provided'
        return np.random.uniform(*self.bin_range[idx][state])

def _standardize_graph(g):
    '''
    Given a graph of time series of tabular data types, standardize the graph dictionary to have
    the same format.
    '''
    data_type = 'time_series'
    gn = {i: [] for i in (list(g.keys()))}
    names = list(g.keys())
    for key, vals in g.items():
        for val in vals:
            val0 = val[0]
            if type(val[0]) not in [list, tuple]:
                data_type = 'tabular'
                val0 = (val[0], 0)
            elif data_type=='tabular':
                raise ValueError(f'The sem dictionary specified has inconsistent parent node types. Found nodes both with and without time lags!')
            gn[key].append([val0, *val[1:]])
    return gn, data_type

def _graph_names2num(g):
    '''
    Given a graph whose keys and items have string names, convert the names to numbers
    '''
    gn = {i: [] for i in range(len(list(g.keys())))}
    names = list(g.keys())
    for key, vals in g.items():
        node_num = names.index(key)
        for val in vals:
            if val[0][0] in names:
                gn[node_num].append([(names.index(val[0][0]), val[0][1]), *val[1:]])
            else:
                raise ValueError(f"Node name '{val[0][0]}' specified as a parent but is not listed in graph keys {names}!")
    return gn

class _Graph():
    def __init__(self, num_nodes): 
        self.graph = {i:[] for i in range(num_nodes)}
        self.num_nodes = num_nodes
  
    def addEdge(self,u,v):
        self.graph[u].append(v) 
  
    def _isCyclic(self, v, visited, visited_during_recusion, adj_graph):
        visited[v] = True
        visited_during_recusion[v] = True

        for child in adj_graph[v]:
            if visited[child] == False:
                if self._isCyclic(child, visited, visited_during_recusion, adj_graph) == True:
                    return True
            elif visited_during_recusion[child] == True:
                return True

        visited_during_recusion[v] = False
        return False

    def isCyclic(self):
        '''
        Check if the causal graph has cycles among the non-lagged connections
        '''
        adj_graph = self.get_adjacency_graph(self.graph)
        visited = [False] * (self.num_nodes + 1)
        visited_during_recusion = [False] * (self.num_nodes + 1)
        for node in range(self.num_nodes):
            if visited[node] == False:
                if self._isCyclic(node,visited,visited_during_recusion, adj_graph) == True:
                    return True
        return False
  
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
        return ad_graph

    def topologicalSort(self):
        '''
        Given a causal graph, return the topologically sorted list of nodes.
        '''
        def sortUtil(graph, n,visited,stack, num_nodes):
            visited[n] = True
            for element in graph[n]:
                if visited[element] == False:
                    sortUtil(graph, element,visited,stack, num_nodes)
            stack.insert(0,n)
        
        adj_graph = self.get_adjacency_graph(self.graph)
        visited = [False]*self.num_nodes
        stack =[]
        for element in range(self.num_nodes):
            if visited[element] == False:
                sortUtil(adj_graph, element, visited,stack, self.num_nodes)
        return list(reversed(stack))