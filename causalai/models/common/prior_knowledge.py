from typing import List, Union, Optional, Dict
from collections import defaultdict
import copy


class PriorKnowledge:
    def __init__(self,
                 forbidden_links: Optional[Dict[Union[int, str], List[Union[int, str]]]] = None,
                 existing_links: Optional[Dict[Union[int, str], List[Union[int, str]]]] = None,
                 root_variables: Optional[List[Union[int, str]]] = None,
                 leaf_variables: Optional[List[Union[int, str]]] = None,
                 forbidden_co_parents: Optional[Dict[Union[int, str], List[Union[int, str]]]] = None,
                 existing_co_parents: Optional[Dict[Union[int, str], List[Union[int, str]]]] = None,
                 fix_co_parents: bool = True,
                 var_names: List[str] = []):
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
        :param forbidden_co_parents: Dictionary of the form {'var3_name': ['var6_name', 'var2_name',...], 'var2_name': ['var4_name',...]}
            Each item in the dictionary denotes that the list of variable values cannot be co-parents of the key
            variable name. In the above example, the first item specifies that var6_name and var2_name cannot
            be co-parents (the cause) of var3_name. If not symmetric, it is expanded to be symmetric. Currently only
            used for Markov Blanket discovery.
        :type forbidden_co_parents: dict, optional
        :param existing_co_parents: Dictionary of the form {'var4_name': ['var1_name', 'var2_name',...], 'var6_name': ['var4_name',...]}
            Each item in the dictionary denotes that the list of variable values must be co-parents of the key
            variable name. In the above example, the first item specifies that var1_name and var2_name must
            be co-parents (the cause) of var4_name. If not symmetric, it is expanded to be symmetric. Currently only
            used for Markov Blanket discovery.
        :type existing_co_parents: dict, optional
        :param fix_co_parents: adds to existing/forbidden co-parents those that are implied by existing/forbidden links
            and by known leaf nodes. Should usually be kept true except in special cases.
        :type fix_co_parents: bool
        :param var_names: Only used if fix_co_parents == True. For instance, var2 is added to forbidden_co_parents[var1]
            whenever var1 in var_names and var2 is in leaf_nodes.
        :type var_names: list
        '''
        self.forbidden_links = copy.deepcopy(forbidden_links)
        self.existing_links = copy.deepcopy(existing_links)
        self.root_variables = copy.deepcopy(root_variables)
        self.leaf_variables = copy.deepcopy(leaf_variables)
        self.forbidden_co_parents = copy.deepcopy(forbidden_co_parents)
        self.existing_co_parents = copy.deepcopy(existing_co_parents)

        if self.forbidden_links is None:
            self.forbidden_links = {}
        if self.existing_links is None:
            self.existing_links = {}
        if self.root_variables is None:
            self.root_variables = []
        if self.leaf_variables is None:
            self.leaf_variables = []
        if self.forbidden_co_parents is None:
            self.forbidden_co_parents = {}
        if self.existing_co_parents is None:
            self.existing_co_parents = {}
        self._sanity_check(fix_co_parents = fix_co_parents, var_names = var_names)

    def _sanity_check(self, fix_co_parents: bool = True, var_names: List[str] = []) -> bool:

        # If fix_co_parents == True, adds to existing/forbidden co-parents those that are implied by existing/forbidden
        # links and by known leaf nodes

        # var_names: if fix_co_parents == True, then var2 is added forbidden_co_parents[var1] whenever
        # var1 in var_names and var2 is in leaf_nodes.

        # Regardless of fix_co_parents, always expands existing/forbidden co-parents via symmetry

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

        # Check for implied existing co-parents by existing_links

        implied_existing_co_parents = defaultdict(set) #implied_existing_co_parents[var]==set of var's implied existing co-parents.

        for parents in self.existing_links.values():
            for parent in parents:
                implied_existing_co_parents[parent].update({x for x in parents if x != parent})

        for v, excluded in self.forbidden_co_parents.items():
            included = implied_existing_co_parents[v]
            contradicting = set(excluded).intersection(included)
            if len(contradicting) > 0:
                raise ValueError(f'The variable {v} is required to have the following variables as co-parents '\
                                 f'according to existing_edges, but this is forbidden by forbidden_co_parents.'\
                                 f'The problematic variables are:' + str(contradicting))

        # If fix_co_parents == True, add the implied existing co-parents to existing_co_parents

        if fix_co_parents:

            for v, added_co_parents in implied_existing_co_parents.items():
                try:
                    current_co_parents = set(self.existing_co_parents[v])
                except KeyError:
                    current_co_parents = set()
                new_entry = list(current_co_parents.union(added_co_parents))
                self.existing_co_parents[v] = new_entry


        # Check for implied forbidden co-parents

        for v in self.leaf_variables:
            for i, co_parents in self.existing_co_parents.items():
                if v in co_parents:
                    raise ValueError(f'The variable {v} is listed both as a leaf and as a co-parent of variable {i}')

        # If fix_co_parents == True, add the implied forbidden co-parents to forbidden_co_parents

        if fix_co_parents:

            # add leaves to forbidden_co_parents (only adding the leaves to values; adding the leaves as keys is done
            # later via the symmetry correction)

            added_forbidden_co_parents = set(self.leaf_variables)

            for var in var_names:
                try:
                    original = set(self.forbidden_co_parents[var])
                except KeyError:
                    original = set()

                self.forbidden_co_parents[var] = list(original.union(added_forbidden_co_parents)- {var})

        # Expand existing/forbidden co-parents via symmetry


        _make_dict_of_lists_symmetric(self.existing_co_parents)
        _make_dict_of_lists_symmetric(self.forbidden_co_parents)


        # Check whether there is any overlap between existing and forbidden co-parents

        for key, existing in self.existing_co_parents.items():
            try:
                forbidden = self.forbidden_co_parents[key]
            except KeyError:
                continue
            flag, intersection = _is_intersection(existing, forbidden)
            if flag:
                raise ValueError(f'The variable {key} has the following variables specified as both existing and\
                 forbidden:' + str(intersection))

    def collect_children(self, target_var: Union[int,str], type: str = 'included'):
        """
        Returns a list of nodes that must be included/excluded as children of target_var from the graph ('transposes'
        the existing_links or forbidden_links dictionaries for the target variable).

        :param target_var: target variable name
        :type target_var: int or str
        :param type: 'included' for variables required to be children of target_var, 'excluded' for variables required not to.
        :type target_var: int or str
        """
        assert type in {'included', 'excluded'}, f'type must be included or excluded'

        links = self.existing_links if type == 'included' else self.forbidden_links
        output = []

        for other_var, parents in links.items():
            if target_var in parents:
                output.append(other_var)

        return output

    def required(self, target_var: Union[int,str], type: str = 'included'):
        """
        Returns a list of variables that must be included/excluded in/from the markov blanket of target_var.

        :param target_var: target variable name
        :type target_var: int or str
        :param type: 'included' for variables required to be in the markov blanket, 'excluded' for variables required to be out of it.
        :type target_var: int or str
        """
        assert type in {'included','excluded'}, f'type must be included or excluded'

        links, co_parents = (self.existing_links, self.existing_co_parents) if type == 'included' else\
                            (self.forbidden_links, self.forbidden_co_parents)
        try:
            parents = links[target_var]
        except KeyError:
            parents = []
        children = self.collect_children(target_var, type)
        try:
            co_parents = co_parents[target_var]
        except KeyError:
            co_parents = []

        if type == 'included':
            return list(set(parents + children + co_parents))
        else:
            return list(set(parents).intersection(set(children), set(co_parents)))


    def isValid(self, parent: Union[int, str], child: Union[int, str]) -> bool:
        '''
        Checks whether a pair of nodes specified as parent-child is valid under the given prior knowledge. Does not take
        co-parents into consideration.

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

    def isValid_co_parent(self, first_co_parent: Union[int, str], second_co_parent: Union[int, str]) -> bool:
        '''
        Checks whether a pair of nodes specified as co-parents is valid under the given prior knowledge.

        :param first_co_parent: First co-parent variable name
        :type first_co_parent: int or str
        :param second_co_parent: Second co-parent variable name
        :type second_co_parent: int or str

        :return: True or False
        :rtype: bool
        '''
        try:
            co_parents = self.forbidden_co_parents[first_co_parent]
        except KeyError:
            co_parents = []
        return not (second_co_parent in co_parents)

    def post_process_tabular(self, graph):
        '''
        Given a causal graph dictionary for tabular data, where keys are children
        and values are parents, this method removes and adds edges depending on the 
        specified prior knowledge in case there are conflicts between the graph and
        the prior knowledge. Does not take co-parents into consideration.

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
                else:
                    g[parent].append(child) # assuming the skeleton is usually more accurate, reverse the edge direction.

        for child in self.existing_links.keys():
            for parent in self.existing_links[child]:
                if parent not in g[child]:
                    g[child].append(parent)
        for key in g.keys():
            g[key] = list(set(g[key]))

        return g


#################### Helper Functions Below #################### 
def _is_intersection(lst1: List, lst2: List):
    intersection = list(set(lst1) & set(lst2))
    flag = len(intersection) > 0
    return flag, intersection

def _make_dict_of_lists_symmetric(dct: dict):
    # Whenever var1 is in dct[var2], adds var2 to dct[var1] (in place).
    to_add_dict = defaultdict(list)
    for key, values in dct.items():
        for value in values:
            to_add_dict[value].append(key)
    for key, to_add_values in to_add_dict.items():
        try:
            dct[key] += to_add_dict[key]
        except KeyError:
            dct[key] = to_add_dict[key]
        dct[key] = list(set(dct[key]))

