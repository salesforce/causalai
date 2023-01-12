import unittest
import numpy as np
import math
from causalai.models.common.prior_knowledge import PriorKnowledge


class TestPriorKnowledge(unittest.TestCase):
    def test_prior_knowledge(self):
        
        # test when a single data array is provided
        forbidden_links = {'C': ['A', 'B'], 'D': ['C']}
        existing_links = {'A': ['B']}
        root_variables = ['A']
        leaf_variables = ['D']
        prior_knowledge = PriorKnowledge(forbidden_links=forbidden_links, 
                                         existing_links=existing_links,
                                         root_variables=root_variables,
                                         leaf_variables=leaf_variables)
        self.assertTrue(not prior_knowledge.isValid('C', 'A'))
        self.assertTrue(not prior_knowledge.isValid('C', 'B'))
        self.assertTrue(not prior_knowledge.isValid('D', 'C'))
        self.assertTrue(prior_knowledge.isValid('A', 'B'))
        self.assertTrue(not prior_knowledge.isValid('B', 'A'))
        self.assertTrue(not prior_knowledge.isValid('D', 'B'))
        self.assertTrue(prior_knowledge.isValid(1, 2))
        

# if __name__ == "__main__":
#     unittest.main()