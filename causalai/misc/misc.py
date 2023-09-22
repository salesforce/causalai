'''
Contains code for graph plotting and computing precision/recall for 
comparing correctness of estimated causal graph with a given ground truth graph.
'''
import networkx as nx
import warnings
import matplotlib.pyplot as plt
import math

def _plot_time_series_graph(graph, filename='', node_size=1000):
    # make sure all variables are present in keys. If missing, create keys with empty list values
    all_vars = []
    for n in graph.keys():
        all_vars.append(n)
        for (m,_) in graph[n]:
            all_vars.append(m)
    all_vars = list(set(all_vars))
    for n in all_vars:
        if n not in graph:
            graph[n] = []

    nvars = len(graph.keys())
    maxlag = 1
    for key in graph.keys():
        for p in graph[key]:
            maxlag = max(maxlag, -p[1])
                
    G=nx.DiGraph()
    for i,n in enumerate(graph.keys()):
        for t in range(maxlag+1):
            xpos = maxlag-t
            ypos = nvars - i -1
            name = f'{n}(t-{t})' if t>0 else f'{n}(t)'
            G.add_node(name,pos=(xpos,ypos))
    
    for t in range(maxlag):
        for n in graph.keys():
            for p,lag in graph[n]:
                lag = -lag
                if t+lag<=maxlag:
                    parent_name = f'{p}(t-{t+lag})' if lag!=0 else f'{p}(t)'
                    child_name = f'{n}(t)' if t==0 else f'{n}(t-{t})'
                    G.add_edge(parent_name,child_name)
    pos=nx.get_node_attributes(G,'pos')
    labels = nx.get_edge_attributes(G,'weight')

    _=nx.draw(G, pos=pos,
            with_labels=True, 
            node_size=node_size, 
            node_color='lightgreen')
    if filename!='':
        plt.savefig(filename)
    else:
        plt.show()


def _plot_tabular_graph(graph, filename='', node_size=1000):
    G = nx.DiGraph()
    for child in graph.keys():
        for parent in graph[child]:
            G.add_edge(parent,child)
    _=nx.draw(G,
            with_labels=True, 
            node_size=node_size, 
            node_color='lightgreen')
    if filename!='':
        plt.savefig(filename)
    else:
        plt.show()

def plot_graph(graph, filename='', node_size=1000):
    '''
    Examples:
    
        Tabular graph:

        g = {'A': ['B', 'C'],
             'B': ['C', 'D'], 'C': []}
        plot_graph(g)

        Time series graph:

        g = {'A': [('B',-1), ('C', -5)],
             'B': [('C', -1), ('D',-3)]}
        plot_graph(g)
    '''
    assert type(graph)==dict, 'graph variable must be a dictionary'
    graph_type = None
    for child in graph.keys():
        assert type(graph[child])==list, f'graph values must be lists, but found {type(graph[child])} {graph[child]}'
        for e in graph[child]:
            if type(e)==tuple:
                graph_type_ = 'time_series'
            elif type(e) in [int, str]:
                graph_type_ = 'tabular'
            else:
                graph_type_ = None
                ValueError(f'The values of all graph keys must be a list of either of type tuple for time series data, or of type str or int for tabular data but found {type(graph[child])}.')
            if graph_type is None:
                graph_type = graph_type_
            else:
                assert graph_type==graph_type_, f'The values of all keys of the variable graph must be of the same type.'
                graph_type = graph_type_
    # all tests passed; plot graph
    if graph_type=='time_series':
        _plot_time_series_graph(graph, filename=filename, node_size=node_size)
    else:
        _plot_tabular_graph(graph, filename=filename, node_size=node_size)



def _get_precision_recall_single(G, G_gt):
    # G and G_gt are both lists of var_names
    if len(G)==0 and len(G_gt)==0:
        return 1.,1.,1.
    if len(G)!=0 and len(G_gt)==0:
        return 0.,1.,0.
    
    p,r,pt,rt=0.,0.,0.,0.
    for i in G:
        pt+=1.
        if i in G_gt:
            p+=1.
    precision = p/(pt+1e-7)
    
    for i in G_gt:
        rt+=1.
        if i in G:
            r+=1.
    recall = r/(rt+1e-7)
    return precision, recall, 2.*precision*recall/(precision+ recall + 1e-7)

def make_symmetric(graph):
    # only valid for tabular graph
    if any([type(p[0]) in [list, tuple] for c,p in graph.items() if len(p)>0]): # if graph is time series, then return graph
        return graph
    g = {key: [] for key in graph.keys()}
    for key in graph.keys():
        for p in graph[key]:
            if p not in g[key]:
                g[key].append(p)
            if key not in g[p]:
                g[p].append(key)
    return g

def get_precision_recall(G, G_gt):
    '''
    Computes the average precision, recall and F1 score
    of the estimated causal graph given the ground truth causal graph
    across variables. Supports both time series and tabular data causal
    graphs.

    :param G: estimated causal graph
    :type G: dict
    :param G_gt: ground truth causal graph
    :type G_gt: dict
    '''
    p,r,f1,t=0.,0.,0.,0.
    for i in G:
        pi, ri, f1i = _get_precision_recall_single(G[i], G_gt[i])
        p+=pi
        r+=ri
        f1+=f1i
        t+=1.
    precision = p/t
    recall = r/t
    f1 = f1/t
    return precision, recall, f1


    return precision, recall, 2.*precision*recall/(precision+ recall + 1e-7)
    
def get_precision_recall_skeleton(G, G_gt):
    '''
    Computes the average precision, recall and F1 score
    of the estimated undirected causal graph given the ground truth directed causal graph
    across variables. Supports tabular data causal graphs only.

    :param G: estimated causal graph
    :type G: dict
    :param G_gt: ground truth causal graph
    :type G_gt: dict
    '''
    if any([type(p[0]) in [list, tuple] for c,p in G_gt.items() if len(p)>0]): # if graph is time series, then return nan
        return math.nan, math.nan, math.nan
    G = make_symmetric(G)
    G_gt = make_symmetric(G_gt)
    p,r,f1,t=0.,0.,0.,0.
    for i in G:
        pi, ri, f1i = _get_precision_recall_single(G[i], G_gt[i])
        p+=pi
        r+=ri
        f1+=f1i
        t+=1.
    precision = p/t
    recall = r/t
    f1 = f1/t
    return precision, recall, f1
