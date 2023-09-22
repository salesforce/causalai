import unittest
import numpy as np
import pandas as pd
from causalai.application import TabularDistributionShiftDetector
from causalai.application.common import distshift_detector_preprocess
class TestTabularDistributionShiftDetector(unittest.TestCase):
    def test_tabular_distribution_shift_detector(self):
        df_normal = pd.read_csv('causalai/tests/datasets/application/tabular_normal.csv')
        df_abnormal = pd.read_csv('causalai/tests/datasets/application/tabular_abnormal.csv')
        T = len(df_normal)
        c_idx = np.array([0]*T + [1]*T)
        
        data_obj, var_names = distshift_detector_preprocess(
            data=[df_normal, df_abnormal],
            domain_index=c_idx,
            domain_index_name='domain_index',
            n_states=2
            )
        
        model = TabularDistributionShiftDetector(
            data_obj=data_obj,
            var_names=var_names,
            domain_index_name='domain_index',
            prior_knowledge=None
            )

        root_causes, graph = model.run(
            pvalue_thres=0.01,
            max_condition_set_size=4,
            return_graph=True
        )
        graph_gt = {'a': {'b', 'c', 'd'},
                    'b': {'a', 'd'},
                    'c': {'a', 'd'},
                    'd': {'a', 'b', 'c'},
                    'domain_index': {'b'}}
        
        self.assertTrue(root_causes=={'b'})
        for key in graph_gt.keys():
            self.assertTrue(graph[key]==graph_gt[key])
        
# if __name__ == "__main__":
#     unittest.main()