from itertools import combinations, permutations
from copy import deepcopy

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

def get_children(graph, p):
    c = []
    for key in graph.keys():
        if p in graph[key]:
            c.append(key)
    return c

def has_both_edges(dag, i, j):
    return i in dag[j] and j in dag[i]

def has_any_edge(dag, i, j):
    return i in dag[j] or j in dag[i]
    
def orient_edges(graph, separation_set, var_names, verbose=False):
    """
    Given an undirected causal graph dictionary where keys are children and values are their corresponding
    parent, a separation_set (dictionary of dictionary) which strores the separatation set for each pair
    of nodes, and var_names, a list of all variable names, this function returns a partially directed graph in the 
    form of a dictionary similar to the input graph dictionary.
    """
    for node in var_names:
        if node not in graph.keys():
            graph[node] = []
    graph = make_symmetric(graph)
    dag = deepcopy(graph) 
    node_list = var_names
    node_pairs = list(permutations(node_list, 2))
    
    # if i-k and j-k and i!-j, and k not in d-separation set of i and j, then i->k<-j. 
    # Note that an incorrect d-sep set can lead to conflicting orientations b/w 2 nodes. 
    # This will lead to the removal of both edges b/w the 2 nodes. We fix this in the code below.
    for (i, j) in node_pairs:
        if not i in graph[j]: # no edge b/w i and j (graph is symmetric)
            for k in set(graph[i]) & set(graph[j]): # k is adjacent to i and j
                if k not in separation_set[i][j] and k not in separation_set[j][i]: # k not in set that separates ij
                    if verbose:
                        print(f'{i}-{k}-{j}, {k} does not d-separate {i}, {j}. Setting {i}->{k}<-{j}')
                    if k in dag[i]:
                        dag[i].remove(k)
                    if k in dag[j]:
                        dag[j].remove(k)
                        
                    # if i not in dag[k]:
                    #     dag[k].append(i)
                    # if j not in dag[k]:
                    #     dag[k].append(j)

    # Fix for the problem described above. Keep the edge b/w 2 nodes unoriented, if conflicting edges were found.
    for i in graph.keys():
        for j in graph.keys():
            if j in graph[i] and not (i in dag[j] and j in dag[i]): # i-j exists, but i and j not adjacent in dag
                dag[i].append(j)
                dag[j].append(i)


    new_change = True
    while new_change: # if the graph changes in the past iteration, more edges could be oriented in the current one.
        new_change=False
        # for each i->k-j, orient edges to k->j
        for (i, j) in node_pairs:
            if not i in dag[j] and not j in dag[i]:
                for k in (set(get_children(dag, i)) - set(dag[i])) & (
                    set(get_children(dag, j)) & set(dag[j])
                ):
                    if j in dag[k]:
                        dag[k].remove(j)
                        new_change = True
        
        # Orient i-j into i->j whenever there is a chain i->k->j.
        for (i, j) in node_pairs:
            if has_both_edges(dag, i, j):
                # Find all nodes k where k is i->k.
                child_i = set()
                for k in get_children(dag, i):
                    if not k in dag[i]: 
                        child_i.add(k)
                # Find nodes j where k->j.
                parent_j = set()
                for k in dag[j]: 
                    if not j in dag[k]: 
                        parent_j.add(k)
                # Check if there is any node k where i->k->j.
                if len(child_i & parent_j) > 0:
                    # Make i-j into i->j
                    dag[i].remove(j)
                    new_change = True

        # for each i-k-j with i->m, j->m, and k-m, orient edges to k->m
        for (i, j) in node_pairs:
            for k in (set(get_children(dag, i)) & set(dag[i]) & set(get_children(dag, j)) & set(dag[j])):
                for m in (
                    (set(get_children(dag, i)) - set(dag[i])) & (set(get_children(dag, j)) - set(dag[j]))
                    & (set(get_children(dag, k)) & set(dag[k])) ):
                    if m in dag[k]:
                        dag[k].remove(m)
                        new_change = True

    for key in dag.keys():
        dag[key] = list(set(dag[key]))
    return dag