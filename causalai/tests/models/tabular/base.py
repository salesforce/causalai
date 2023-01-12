import unittest
import numpy as np
import math
import pickle as pkl

from causalai.models.common.prior_knowledge import PriorKnowledge
from causalai.data.tabular import TabularData
from causalai.data.transforms.tabular import StandardizeTransform
from causalai.models.tabular.base import BaseTabularAlgo

class TestBaseTabularModel(unittest.TestCase):
    def test_base_tabular_model(self):
        
        with open('causalai/tests/datasets/tabular/synthetic_data1.pkl', 'rb') as f:
            dataset = pkl.load(f)
        graph_gt, data_array = dataset['graph'], dataset['data']
        var_names = list(graph_gt.keys())
        data_train = data_array

        StandardizeTransform_ = StandardizeTransform()
        StandardizeTransform_.fit(data_train)

        data_train_trans = StandardizeTransform_.transform(data_train)

        data_train_obj = TabularData(data_train_trans, var_names=var_names)
        prior_knowledge = PriorKnowledge(forbidden_links={'a': ['b']}) # b cannot be a parent of a

        model = BaseTabularAlgo(data=data_train_obj, prior_knowledge=prior_knowledge, use_multiprocessing=False)


        # check if get_all_parents() returns the correct set of parents
        all_parents = model.get_all_parents(target_var='a')
        all_parents_gt = ['b', 'c', 'd', 'e', 'f', 'g']
        self.assertTrue(set(all_parents)==set(all_parents_gt))

        # check if get_candidate_parents() returns the correct set of parents
        candidate_parents = model.get_candidate_parents(target_var='a')
        candidate_parents_gt = ['c', 'd', 'e', 'f', 'g']
        self.assertTrue(set(candidate_parents)==set(candidate_parents_gt))

        # check if sort_parents() sorts the parents correctly based on pvalues
        parents_vals = {'a': 0.0, 'c': 0.05}
        sorted_parent = model.sort_parents(parents_vals)
        self.assertTrue(sorted_parent==['c', 'a'])


# if __name__ == "__main__":
#     unittest.main()
