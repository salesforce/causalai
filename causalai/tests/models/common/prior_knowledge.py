import unittest
import numpy as np
import math
from causalai.models.common.prior_knowledge import _make_dict_of_lists_symmetric
from causalai.models.common.prior_knowledge import PriorKnowledge

class Test_make_dict_of_lists_symmetric(unittest.TestCase):

    def test_make_dict_of_lists_symmetric(self):
        dict1 = {}
        result1 = {}
        dict2 = {0: [1, 3, 4, 6], 1: [4, 6], 3: [5, 7], 4: [1, 3, 7], 6: [0, 3, 5], 7: []}
        result2 = {0: [1, 3, 4, 6], 1: [0, 4, 6], 3: [0, 4, 5, 6, 7], 4: [0, 1, 3, 7], 5: [3, 6], 6: [0, 1, 3, 5], 7: [3, 4]}
        _make_dict_of_lists_symmetric(dict1)
        _make_dict_of_lists_symmetric(dict2)
        for x in dict2.values():
            x.sort()
        self.assertEqual(dict1, result1)
        self.assertEqual(dict2, result2)

class TestPriorKnowledge(unittest.TestCase):
    def test_prior_knowledge(self):
        forbidden_links = {'C': ['A', 'B'], 'D': ['C']}
        existing_links = {'B': ['A']}
        root_variables = ['A']
        leaf_variables = ['D']
        prior_knowledge = PriorKnowledge(forbidden_links=forbidden_links, 
                                         existing_links=existing_links,
                                         root_variables=root_variables,
                                         leaf_variables=leaf_variables)
        self.assertTrue(not prior_knowledge.isValid('C', 'A'))
        self.assertTrue(prior_knowledge.isValid('C', 'B'))
        self.assertTrue(not prior_knowledge.isValid('D', 'C'))
        self.assertTrue(prior_knowledge.isValid('A', 'B'))
        self.assertTrue(not prior_knowledge.isValid('B', 'A'))
        self.assertTrue(not prior_knowledge.isValid('D', 'B'))

    def test_existing_links_forbidden_co_parents_collision(self):
        existing_links = {3: [1, 2]}
        forbidden_co_parents = {2: [1]}
        self.assertRaises(ValueError, PriorKnowledge, existing_links = existing_links, forbidden_co_parents = forbidden_co_parents)

    def test_prior_knowledge_existing_co_parents_expansions_via_existing_links(self):
        existing_links = {0: [1, 2, 3], 2: [1, 4], 5: [1, 6, 7]}
        existing_co_parents = {0: []}
        expected_existing_co_parents = {0: [], 1: [2, 3, 4, 6, 7], 2: [1, 3], 3: [1, 2], 4: [1], 6: [1, 7], 7: [1, 6]}
        output = PriorKnowledge(existing_links=existing_links, existing_co_parents=existing_co_parents,
                                 fix_co_parents=True)
        for value in output.existing_co_parents.values():
            value.sort()
        self.assertEqual(expected_existing_co_parents, output.existing_co_parents)
        expected_existing_co_parents = {0: []}
        output = PriorKnowledge(existing_links=existing_links, existing_co_parents=existing_co_parents,
                                fix_co_parents=False)

        for value in output.existing_co_parents.values():
            value.sort()
        self.assertEqual(expected_existing_co_parents, output.existing_co_parents)

    def test_prior_knowledge_leaves_existing_co_parents_collision(self):
        leaf_variables = [0]
        existing_co_parents = {1: [0]}
        self.assertRaises(ValueError, PriorKnowledge, leaf_variables = leaf_variables,
                          existing_co_parents = existing_co_parents, fix_co_parents = True)
        self.assertRaises(ValueError, PriorKnowledge, leaf_variables=leaf_variables,
                          existing_co_parents=existing_co_parents, fix_co_parents = False)
        existing_co_parents = {0: [1, 2, 3], 2: [1, 4], 5: [0, 1, 6, 7]}
        self.assertRaises(ValueError, PriorKnowledge, leaf_variables=leaf_variables,
                          existing_co_parents=existing_co_parents, fix_co_parents=True)
        self.assertRaises(ValueError, PriorKnowledge, leaf_variables=leaf_variables,
                          existing_co_parents=existing_co_parents, fix_co_parents=False)

    def test_prior_knowledge_existing_co_parents_expansions_via_existing_links(self):

        leaf_variables = [0, 1, 8]
        forbidden_co_parents = {0: [1, 2, 3], 2: [1, 4], 5: [1, 6, 7]}
        var_names = list(range(9))
        expected_forbidden_co_parents = {0: [1, 2, 3, 4, 5, 6, 7, 8], 1: [0, 2, 3, 4, 5, 6, 7, 8], 2: [0, 1, 4, 8],
                                         3: [0, 1, 8], 4: [0, 1, 2, 8], 5: [0, 1, 6, 7, 8], 6: [0, 1, 5, 8],
                                         7: [0, 1, 5, 8], 8: [0, 1, 2, 3, 4, 5, 6, 7]}
        output = PriorKnowledge(leaf_variables=leaf_variables, forbidden_co_parents=forbidden_co_parents,
                                fix_co_parents=True, var_names=var_names)
        for value in output.forbidden_co_parents.values():
            value.sort()
        self.assertEqual(expected_forbidden_co_parents, output.forbidden_co_parents)
        expected_forbidden_co_parents = {0: [1, 2, 3], 1: [0, 2, 5], 2: [0, 1, 4], 3: [0], 4: [2], 5: [1, 6, 7], 6: [5],
                                         7: [5]}
        output = PriorKnowledge(leaf_variables=leaf_variables, forbidden_co_parents=forbidden_co_parents,
                                fix_co_parents=False, var_names=var_names)
        for value in output.forbidden_co_parents.values():
            value.sort()
        self.assertEqual(expected_forbidden_co_parents, output.forbidden_co_parents)
        var_names = []
        output = PriorKnowledge(leaf_variables=leaf_variables, forbidden_co_parents=forbidden_co_parents,
                                fix_co_parents=False, var_names=var_names)
        for value in output.forbidden_co_parents.values():
            value.sort()
        self.assertEqual(expected_forbidden_co_parents, output.forbidden_co_parents)
        var_names = [2, 7, 8]
        expected_forbidden_co_parents = {0: [1, 2, 3, 7, 8], 1: [0, 2, 5, 7, 8], 2: [0, 1, 4, 8], 3: [0], 4: [2],
                                         5: [1, 6, 7], 6: [5], 7: [0, 1, 5, 8], 8: [0, 1, 2, 7]}
        output = PriorKnowledge(leaf_variables=leaf_variables, forbidden_co_parents=forbidden_co_parents,
                                fix_co_parents=True, var_names=var_names)
        for value in output.forbidden_co_parents.values():
            value.sort()
        self.assertEqual(expected_forbidden_co_parents, output.forbidden_co_parents)

    def test_prior_knowledge_existing_forbidden_co_parents_collision(self):
        existing_co_parents = {3: [1, 2]}
        forbidden_co_parents = {2: [3]}
        self.assertRaises(ValueError, PriorKnowledge, existing_co_parents=existing_co_parents,
                          forbidden_co_parents=forbidden_co_parents)
        forbidden_co_parents = {2: [1]}
        PriorKnowledge(existing_co_parents=existing_co_parents, forbidden_co_parents=forbidden_co_parents)

    def test_collect_children(self):
        links = {2: [0, 1], 3: [1, 2], 4: [2, 3], 5: [3], 6: [2, 7, 8]}
        prior_knowledge = PriorKnowledge(existing_links=links)
        results = {0: [2], 1: [2, 3], 2: [3, 4, 6], 3: [4, 5], 4: [], 5: [], 6: [], 7: [6], 8: [6]}
        for var in range(9):
            children = sorted(prior_knowledge.collect_children(target_var = var, type = 'included'))
            self.assertEqual((var, results[var]), (var, children))
        prior_knowledge = PriorKnowledge(forbidden_links=links)
        for var in range(9):
            children = sorted(prior_knowledge.collect_children(target_var=var, type='excluded'))
            self.assertEqual((var, results[var]), (var, children))

    def test_required(self):
        links = {2: [0, 1], 3: [1, 2], 4: [2, 3], 5: [2], 6: [2, 7, 8]}
        prior_knowledge = PriorKnowledge(existing_links=links)
        results = {0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 3, 4, 5, 6, 7, 8], 3: [1, 2, 4], 4: [2, 3], 5: [2], 6: [2, 7, 8], 7: [2, 6, 8], 8: [2, 6, 7]}
        for var in range(9):
            result = results[var]
            mb = sorted(prior_knowledge.required(target_var=var, type='included'))
            self.assertEqual((var, result), (var, mb))
        links = {0: [1, 2, 3, 4]}
        forbidden_co_parents = {0: [2, 3]}
        prior_knowledge = PriorKnowledge(forbidden_links=links, forbidden_co_parents = forbidden_co_parents)
        result = []
        not_mb = sorted(prior_knowledge.required(target_var=0, type='excluded'))
        self.assertEqual((0, result), (0, not_mb))
        links = {0: [1, 2, 3, 4], 1: [0, 2], 2: [0], 3: [0,2], 4: [0, 1, 2, 3]}
        prior_knowledge = PriorKnowledge(forbidden_links=links, forbidden_co_parents=forbidden_co_parents)
        result = [2, 3]
        not_mb = sorted(prior_knowledge.required(target_var=0, type='excluded'))
        self.assertEqual((0, result), (0, not_mb))

    def test_isValid_co_parent(self):
        forbidden_co_parents = {0: [1,3], 3: [4], 4: [3]}
        prior_knowledge = PriorKnowledge(forbidden_co_parents=forbidden_co_parents)
        self.assertFalse(prior_knowledge.isValid_co_parent(0,1))
        self.assertFalse(prior_knowledge.isValid_co_parent(0, 3))
        self.assertFalse(prior_knowledge.isValid_co_parent(1, 0))
        self.assertFalse(prior_knowledge.isValid_co_parent(3, 0))
        self.assertFalse(prior_knowledge.isValid_co_parent(3, 4))
        self.assertFalse(prior_knowledge.isValid_co_parent(4, 3))

        self.assertTrue(prior_knowledge.isValid_co_parent(0, 2))
        self.assertTrue(prior_knowledge.isValid_co_parent(0, 4))
        self.assertTrue(prior_knowledge.isValid_co_parent(4, 0))
        self.assertTrue(prior_knowledge.isValid_co_parent(1, 3))
        self.assertTrue(prior_knowledge.isValid_co_parent(3, 1))
        self.assertTrue(prior_knowledge.isValid_co_parent(1, 2))
        self.assertTrue(prior_knowledge.isValid_co_parent(2, 1))
        self.assertTrue(prior_knowledge.isValid_co_parent(2, 4))
        self.assertTrue(prior_knowledge.isValid_co_parent(4, 2))


#if __name__ == "__main__":
#     unittest.main()