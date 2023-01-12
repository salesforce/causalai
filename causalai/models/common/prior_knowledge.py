from typing import List, Union, Optional, Dict


class PriorKnowledge:
    def __init__(self,
                 forbidden_links: Optional[Dict[Union[int, str], List[Union[int, str]]]] = None,
                 existing_links: Optional[Dict[Union[int, str], List[Union[int, str]]]] = None,
                 root_variables: Optional[List[Union[int, str]]] = None,
                 leaf_variables: Optional[List[Union[int, str]]] = None):
        '''
        This class allows adding any prior knoweledge to the causal discovery process by either
        forbidding links that are known to not exist, or adding back links that do exist
        based on expert knowledge. This class can be used to specify prior knowledge for
        both tabular and time series data. Note that for both data types, prior knowledge 
        is specified only in terms of variable names, and no time steps need to be specified 
        in the case of time series. 

        :param forbidden_links: Dictionary of the form {'var3_name': ['var6_name', 'var2_name',...], 'var2_name': ['var4_name',...]}
            Each item in the dictionary denotes that the list of variable values cannot be parents of the key
            variable name. In the above example, the first item specifies that var6_name and var2_name cannot 
            be parents (the cause) of var3_name.
        :type forbidden_links: dict, optional
        :param existing_links: Dictionary of the form {'var4_name': ['var1_name', 'var2_name',...], 'var6_name': ['var4_name',...]}
            Each item in the dictionary denotes that the list of variable values must be parents of the key
            variable name. In the above example, the first item specifies that var1_name and var2_name must 
            be parents (the cause) of var4_name.
        :type existing_links: dict, optional
        :param root_variables: List of the form ['var7_name',...]
            Any variable specified in this list says that it do not have any parents (incoming causal links).
            Note that this information can alternatively be specified in the forbidden_links argument by listing all the
            variables in the dataset for the key variable var7_name.
        :type root_variables: list, optional
        :param leaf_variables: List of the form ['var7_name',...]
            Any variable specified in this list says that it do not have any children (outgoing causal links).
        :type leaf_variables: list, optional
        '''
        self.forbidden_links = forbidden_links
        self.existing_links = existing_links
        self.root_variables = root_variables
        self.leaf_variables = leaf_variables

        if self.forbidden_links is None:
            self.forbidden_links = {}
        if self.existing_links is None:
            self.existing_links = {}
        if self.root_variables is None:
            self.root_variables = []
        if self.leaf_variables is None:
            self.leaf_variables = []
        self._sanity_check()

    def _sanity_check(self) -> bool:

        # parents specified in existing_links cannot exist if the key variable is specified in root_variables
        # parents specified in existing_links must not conflict with forbidden_links
        all_children = []
        all_parents = []
        for v in self.existing_links.keys():
            all_children.append(v)
            all_parents.extend(self.existing_links[v])
            if v in self.forbidden_links:
                flag, intersection = _is_intersection(self.forbidden_links[v], self.existing_links[v])
                if flag:
                    raise ValueError(
                        f'The variable {v} is specified as a child of node(s) {intersection} in the argument existing_links,' \
                        f' but {v} is also specified as a forbidden child of node(s) {intersection} in forbidden_links.')
        for v in self.root_variables:
            if v in all_children:
                raise ValueError(f'The variable {v} is specified as a child in the argument existing_links,' \
                           f' but {v} is also specified as a root_variable which cannot have parents.')

        for v in self.leaf_variables:
            if v in all_parents:
                raise ValueError(f'The variable {v} is specified as a parent in the argument existing_links,' \
                           f' but {v} is also specified as a leaf_variable which cannot have children.')

        # make sure the existing_links is not creating any cycle
        # if Graph(self.existing_links).isCyclic():
        #     ValueError(f'The links specified in existing_links is creating a cyclic graph. Only DAG is allowed for causal discovery.')

    def isValid(self, parent: Union[int, str], child: Union[int, str]) -> bool:
        '''
        Checks whether a pair of nodes specified as parent-child is valid under the given prior knowledge.

        :param parent: Parent variable name
        :type parent: int or str
        :param child: Child variable name
        :type child: int or str

        :return: True or False
        :rtype: bool
        '''
        if child in self.root_variables:
            return False
        if child in self.forbidden_links and parent in self.forbidden_links[child]:
            return False
        if parent in self.existing_links and child in self.existing_links[parent]: 
            # if the argument parent->child is specified as child->parent in existing_links, then return False
            return False
        if parent in self.leaf_variables:
            return False
        return True

    def post_process_tabular(self, graph):
        '''
        Given a causal graph dictionary for tabular data, where keys are children
        and values are parents, this method removes and adds edges depending on the 
        specified prior knowledge in case their are conflicts between the graph and
        the prior knowledge.

        :param graph: causal graph dictionary as explained above
        :type graph: dict

        :return: causal graph dictionary with prior knowledge enforced
        :rtype: dict
        '''
        g = {key: [] for key in graph.keys()}
        for child in graph.keys():
            for parent in graph[child]:
                if self.isValid(parent, child):
                    g[child].append(parent)

        for child in self.existing_links.keys():
            for parent in self.existing_links[child]:
                if parent not in g[child]:
                    g[child].append(parent)
        return g


#################### Helper Functions Below #################### 
def _is_intersection(lst1: List, lst2: List):
    intersection = list(set(lst1) & set(lst2))
    flag = len(intersection) > 0
    return flag, intersection
