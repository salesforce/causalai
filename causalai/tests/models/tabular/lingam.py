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
from causalai.models.tabular.lingam import LINGAM

class TestLINGAMTabularModel(unittest.TestCase):
    def test_lingam_tabular_model(self):
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

        model = LINGAM(
                data=data_train_obj,
                prior_knowledge=prior_knowledge,
                use_multiprocessing=False
                )

        results = model.run(pvalue_thres=0.05)
        graph_gt =  {
                'a': {'value_dict': {'a': 0.0, 'b': 0.06801337712178565, 'c': 0.0, 
                                     'd': 0.0, 'e': 0.0, 'f': 0.0, 'g': 0.0}, 
                      'pvalue_dict': {'a': 1.0, 'b': 2.5622588569330236e-06, 'c': 1.0, 
                                      'd': 1.0, 'e': 1.0, 'f': 1.0, 'g': 1.0}, 
                      'parents': ['b']}, 
                'b': {'value_dict': {'a': 0.0, 'b': 0.0, 'c': 0.12796351759849509, 
                                     'd': 0.10546817784255101, 'e': 0.0, 'f': 0.08825588181436207, 'g': 0.0}, 
                      'pvalue_dict': {'a': 1.0, 'b': 1.0, 'c': 1.40500866515624e-19, 'd': 7.01178423364487e-14, 
                                      'e': 1.0, 'f': 4.081074099604285e-10, 'g': 1.0}, 
                      'parents': ['c', 'd', 'f']}, 
                'c': {'value_dict': {'a': 0.0, 'b': 0.0, 'c': 0.0, 'd': 0.0, 'e': 0.0, 'f': 0.0, 'g': 0.0}, 
                      'pvalue_dict': {'a': 1.0, 'b': 1.0, 'c': 1.0, 'd': 1.0, 'e': 1.0, 'f': 1.0, 'g': 1.0}, 
                      'parents': []}, 
                'd': {'value_dict': {'a': 0.0, 'b': 0.0, 'c': 0.0, 'd': 0.0, 'e': 0.0, 'f': 0.0, 'g': 0.0}, 
                      'pvalue_dict': {'a': 1.0, 'b': 1.0, 'c': 1.0, 'd': 1.0, 'e': 1.0, 'f': 1.0, 'g': 1.0}, 
                      'parents': []}, 
                'e': {'value_dict': {'a': 0.0, 'b': 0.0, 'c': 0.0, 'd': 0.0, 'e': 0.0, 
                                     'f': 0.097299238339312, 'g': 0.0}, 
                      'pvalue_dict': {'a': 1.0, 'b': 1.0, 'c': 1.0, 'd': 1.0, 'e': 1.0, 
                                      'f': 1.163072880694903e-11, 'g': 1.0}, 
                      'parents': ['f']}, 
                'f': {'value_dict': {'a': 0.0, 'b': 0.0, 'c': 0.10509976602734823, 'd': 0.0, 
                                     'e': 0.0, 'f': 0.0, 'g': 0.0}, 
                      'pvalue_dict': {'a': 1.0, 'b': 1.0, 'c': 2.1316724714250732e-13, 'd': 1.0, 
                                      'e': 1.0, 'f': 1.0, 'g': 1.0}, 
                      'parents': ['c']}, 
                'g': {'value_dict': {'a': 0.0, 'b': 0.0, 'c': 0.0, 'd': 0.10984150690850368, 
                                     'e': 0.0, 'f': 0.0, 'g': 0.0}, 
                      'pvalue_dict': {'a': 1.0, 'b': 1.0, 'c': 1.0, 'd': 1.5101658034156483e-14, 
                                      'e': 1.0, 'f': 1.0, 'g': 1.0}, 
                      'parents': ['d']}
                    }
        for key in graph_gt.keys():
            self.assertTrue(graph_gt[key]==results[key])


# test = TestLINGAMTabularModel()
# test.test_lingam_tabular_model()

        