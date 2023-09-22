import networkx as nx
import numpy as np
from typing import Union
from collections import defaultdict
import random
from ...models.common.prior_knowledge import PriorKnowledge

def causalai2networkx(graph: dict) -> nx.DiGraph:
    #Takes a dictionary of the form graph[var]=list of parents of var, converts to the equivalent networkx DiGraph.
    nodes = []
    edges = []
    for var, parents in graph.items():
        nodes.append(var)
        for parent in parents:
            edges.append((parent, var))
    output = nx.DiGraph()
    output.add_nodes_from(nodes)
    output.add_edges_from(edges)
    return output

def networkx2causalai(graph: nx.DiGraph) -> dict:
    #Reverse conversion
    output = defaultdict(list)
    for var1, var2 in graph.edges():
        output[var2].append(var1)
    return dict(output)

def collect_all_knowledge(graph: nx.DiGraph) -> dict:
    #Collects all information about existing/forbidden links and co-parents, and root/leaf variables, from the graph
    existing_links = networkx2causalai(graph)
    forbidden_links = defaultdict(list)
    root_variables = []
    leaf_variables = []
    existing_co_parents = defaultdict(list)
    forbidden_co_parents = defaultdict(list)
    all_vars = list(graph.nodes)
    for index, var1 in enumerate(all_vars):
        for var2 in (all_vars[:index]+all_vars[index+1:]):
            if not graph.has_edge(var2, var1):
                forbidden_links[var1].append(var2)

    for var in all_vars:
        parents = list(graph.predecessors(var))
        if len(parents)==0:
            root_variables.append(var)
        children = list(graph.successors(var))
        if len(children) == 0:
            leaf_variables.append(var)

    all_parents = [list(graph.predecessors(var)) for var in all_vars]
    for index, var1 in enumerate(all_vars):
        for var2 in (all_vars[:index]+all_vars[index+1:]):
            are_co_parents = False
            for parent_collection in all_parents:
                if (var1 in parent_collection) and (var2 in parent_collection):
                    existing_co_parents[var1].append(var2)
                    existing_co_parents[var2].append(var1)
                    are_co_parents = True
                    break
            if not are_co_parents:
                forbidden_co_parents[var1].append(var2)
                forbidden_co_parents[var2].append(var1)
    return {'existing_links': existing_links, 'forbidden_links': dict(forbidden_links), 'root_variables': root_variables,
            'leaf_variables': leaf_variables, 'existing_co_parents': dict(existing_co_parents),
            'forbidden_co_parents': dict(forbidden_co_parents)}

def get_partial_knowledge(graph: nx.DiGraph, full_knowledge: dict, inclusion_probability: float = 0.1):
    #Randomly drops information from the input
    new_dicts = []
    for old_dict in [full_knowledge['existing_links'], full_knowledge['forbidden_links'], full_knowledge['existing_co_parents'],
                     full_knowledge['forbidden_co_parents']]:
        new_dict = defaultdict(list)
        for key, values in old_dict.items():
            for value in values:
                if random.uniform(0,1) < inclusion_probability:
                    new_dict[key].append(value)
        new_dicts.append(dict(new_dict))

    new_root_variables = [x for x in full_knowledge['root_variables'] if random.uniform(0,1) < inclusion_probability]
    new_leaf_variables = [x for x in full_knowledge['leaf_variables'] if random.uniform(0,1) < inclusion_probability]

    var_names = list(graph.nodes)

    return PriorKnowledge(existing_links = new_dicts[0], forbidden_links = new_dicts[1],
                          existing_co_parents = new_dicts[2], forbidden_co_parents = new_dicts[3],
                          leaf_variables = new_leaf_variables, root_variables = new_root_variables,
                          fix_co_parents = True, var_names=var_names)

def compute_markov_blanket(graph: nx.DiGraph, var: int):
    '''
    Computes the true minimal markov blanket of var in graph (parents, children, co-parents).
    '''
    var_children=set(graph.successors(var))
    var_parents=set(graph.predecessors(var))
    co_parents=set()
    for other_var in graph.nodes:
        if other_var != var:
            other_var_children = set(graph.successors(other_var))
            if len(var_children.intersection(other_var_children))>0:
                co_parents.add(other_var)
    return var_children.union(var_parents).union(co_parents)

class PerfectTest(object):
    '''
    A perfect CI test (assuming faithfulness, using d-separation), with access to the underlying graph.
    '''
    def __init__(self, graph: nx.digraph, full_data: np.array):
        self.graph = graph
        self.full_data = full_data
        self.counter=0
    def run_test(self, x_data: np.array, y_data: np.array, z_data: np.array):
        self.counter += 1
        z=[]
        for column_index in range(self.full_data.data_arrays[0].shape[1]):
            column = self.full_data.data_arrays[0][:,column_index]
            if all(x_data == column):
                x=self.full_data.index2var_name(column_index)

            elif all(y_data == column):
                y=self.full_data.index2var_name(column_index)

            elif z_data is not None:
                for z_column_index in range(z_data.shape[1]):
                    if all(z_data[:,z_column_index]==column):
                        z.append(self.full_data.index2var_name(column_index))


        p_value=1.0 if nx.d_separated(self.graph,set([self.full_data.index2var_name(x)]),set([self.full_data.index2var_name(y)]),set([self.full_data.index2var_name(i) for i in z])) else 0.0
        return [None,p_value]







