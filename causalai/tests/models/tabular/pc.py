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
from causalai.models.tabular.pc import PCSingle, PC

class TestPCTabularModel(unittest.TestCase):
    def test_pc_tabular_model(self):
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


        # check the outputs of Full Causal Discovery methods
        prior_knowledge = None

        pc = PC(
                data=data_train_obj,
                prior_knowledge=prior_knowledge,
                use_multiprocessing=False
                )

        results = pc.run(pvalue_thres=0.05)
        graph_gt = {
                'a': {'parents': ['b'], 
                      'value_dict': {'b': 0.07078601437557672}, 
                      'pvalue_dict': {'b': 5.446967467574575e-07}}, 
                'b': {'parents': ['a', 'c', 'a', 'c', 'd', 'f'], 
                      'value_dict': {'a': 0.07078601437557672, 'c': 0.12793460296040382, 
                                     'd': 0.10854396571591053, 'f': 0.0908882373054033}, 
                      'pvalue_dict': {'a': 5.446967467574575e-07, 'c': 1.098858342879257e-19, 
                                      'd': 1.420426169531583e-14, 'f': 1.2169552268898473e-10}}, 
                'c': {'parents': ['b', 'f'], 
                      'value_dict': {'b': 0.13073702578458776, 'f': 0.09704511418059478}, 
                      'pvalue_dict': {'b': 1.6769609624725097e-20, 'f': 6.157478397885461e-12}}, 
                'd': {'parents': ['g', 'b', 'g'], 
                      'value_dict': {'b': 0.11092999619152272, 'g': 0.1089284982478336}, 
                      'pvalue_dict': {'b': 3.677015297609073e-15, 'g': 1.1389090834247264e-14}}, 
                'e': {'parents': ['f'], 
                      'value_dict': {'f': 0.10041127537290304}, 
                      'pvalue_dict': {'f': 1.108105918056936e-12}}, 
                'f': {'parents': ['c', 'e', 'b', 'c', 'e'], 
                      'value_dict': {'b': 0.09303108395795855, 'c': 0.09700857068785299, 
                                     'e': 0.10041127537290304}, 
                      'pvalue_dict': {'b': 4.389859904079807e-11, 'c': 6.301196424981861e-12, 
                                      'e': 1.108105918056936e-12}}, 
                'g': {'parents': ['d'], 
                      'value_dict': {'d': 0.10984150716124015}, 
                      'pvalue_dict': {'d': 6.7754786942117675e-15}}}

        for key in graph_gt.keys():
            self.assertTrue(results[key]==graph_gt[key])


# if __name__ == "__main__":
#     unittest.main()
