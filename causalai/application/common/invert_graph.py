from typing import List, Dict

def invert_graph_and_remove_duplicates(
    graph: Dict[str, List[str]]
    ):
    '''
    Invert the directed acyclic graph.
    :param graph: DAG graph of the causal relationships
    :type graph: Dict[str, List[str]]
    :return: inverted graph
    :type: Dict[str, List[str]]
    '''
    inv_graph = {n: set() for n in graph.keys()}
    for key in graph.keys():
        parents = graph[key]
        for parent in parents:
            inv_graph[parent].add(key)
    return inv_graph     