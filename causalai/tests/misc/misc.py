import unittest
import numpy as np
import math
from causalai.data.tabular import TabularData

from causalai.misc.misc import get_precision_recall


class TestMisc(unittest.TestCase):
    def test_precision_recall(self):
        graph_est = {'a': [],
                     'b': ['a', 'c'],
                     'c': [],
                     'd': ['g'],
                     'e': [],
                     'f': ['c', 'e'],
                     'g': []}
        graph_gt = {'a': [],
                     'b': ['a', 'f'],
                     'c': ['b', 'f'],
                     'd': ['b', 'g'],
                     'e': ['f'],
                     'f': [],
                     'g': []}
        p,r,f1 = get_precision_recall(graph_est, graph_gt)
        print(p,r,f1)
        self.assertTrue(p==0.49999998214285873)
        self.assertTrue(r==0.5714285642857146)
        self.assertTrue(f1==0.4523809289682556)
        
# if __name__ == "__main__":
#     unittest.main()