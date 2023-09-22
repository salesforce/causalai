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
from causalai.models.tabular.ges import GES

class TestGESTabularModel(unittest.TestCase):
    def test_ges_tabular_model(self):
        with open(f'causalai/tests/datasets/tabular/synthetic_data1.pkl', 'rb') as f:
            dataset = pkl.load(f)
        graph_gt, data_array = dataset['graph'], dataset['data']
        var_names = list(graph_gt.keys())

        data_train = data_array

        StandardizeTransform_ = StandardizeTransform()
        StandardizeTransform_.fit(data_train)

        data_train_trans = StandardizeTransform_.transform(data_train)

        data_train_obj = TabularData(data_train_trans, var_names=var_names)
        prior_knowledge = PriorKnowledge(forbidden_links={'a': ['b']}) # b cannot be a parent of a


        # check the outputs of Full Causal Discovery methods
        prior_knowledge = None

        model = GES(
                data=data_train_obj,
                prior_knowledge=prior_knowledge,
                use_multiprocessing=False
                )

        results = model.run(pvalue_thres=0.05)
        graph_gt =  {'a': {'value_dict': {}, 'pvalue_dict': {}, 'parents': []},
                     'b': {'value_dict': {}, 'pvalue_dict': {}, 'parents': ['a', 'd']},
                     'c': {'value_dict': {}, 'pvalue_dict': {}, 'parents': ['b', 'f']},
                     'd': {'value_dict': {}, 'pvalue_dict': {}, 'parents': ['g']},
                     'e': {'value_dict': {}, 'pvalue_dict': {}, 'parents': []},
                     'f': {'value_dict': {}, 'pvalue_dict': {}, 'parents': ['b', 'e']},
                     'g': {'value_dict': {}, 'pvalue_dict': {}, 'parents': ['d']}}
        for key in graph_gt.keys():
            self.assertTrue(graph_gt[key]==results[key])
        

# test = TestGESTabularModel()
# test.test_ges_tabular_model()
