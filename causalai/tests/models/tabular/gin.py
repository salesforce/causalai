import unittest
import numpy as np
import math
import pickle as pkl
try:
    import ray
except:
    pass

from causalai.models.common.prior_knowledge import PriorKnowledge
from causalai.data.tabular import TabularData
from causalai.data.transforms.tabular import StandardizeTransform
from causalai.models.tabular.gin import GIN

class TestGINTabularModel(unittest.TestCase):
    def test_gin_tabular_model(self):
        sample_size = 500
        np.random.seed(2)
        np.random.seed(2)
        L0 = np.random.uniform(-1, 1, size=sample_size)
        L1 = np.random.uniform(1., 2.) * L0 + np.random.uniform(-1, 1, size=sample_size)
        L2 = np.random.uniform(1., 2.) * L0 + np.random.uniform(1.2, 1.8) * L1 + np.random.uniform(-1, 1, size=sample_size)
        a = np.random.uniform(1., 2.) * L0 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        b = np.random.uniform(1., 2.) * L0 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        c = np.random.uniform(1., 2.) * L0 + 0.2 * np.random.uniform(-1, 1, size=sample_size) 
        d = np.random.uniform(1., 2.) * L1 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        e = np.random.uniform(1., 2.) * L1 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        f = np.random.uniform(1., 2.) * L1 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        g = np.random.uniform(1., 2.) * L2 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        h = np.random.uniform(1., 2.) * L2 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        i = np.random.uniform(1., 2.) * L2 + 0.2 * np.random.uniform(-1, 1, size=sample_size)
        data = np.array([a, b, c, d, e, f, g, h, i]).T

        ground_truth = [['a','b','c'], ['d','e','f'], ['g','h','i']]

        StandardizeTransform_ = StandardizeTransform()
        StandardizeTransform_.fit(data)

        data_trans = StandardizeTransform_.transform(data)

        data_obj = TabularData(data_trans, var_names=['a','b','c','d','e','f','g','h','i'])

        model = GIN(data_obj, prior_knowledge=None)
        result = model.run(pvalue_thres=0.05)

        causal_order_sets = [set(i) for i in model.causal_order]

        for l in ground_truth:
            self.assertTrue(set(l) in causal_order_sets)
        
# test = TestGINTabularModel()
# test.test_gin_tabular_model()