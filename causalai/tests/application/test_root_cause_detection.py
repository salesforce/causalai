import unittest
import numpy as np
import pandas as pd
from causalai.application import RootCauseDetector
from causalai.application.common import rca_preprocess
class TestMicroServiceRootCauseDetection(unittest.TestCase):
    def test_microservice_root_cause_detection(self):
        df_normal = pd.read_csv('causalai/tests/datasets/application/cloud_normal.csv')
        df_abnormal = pd.read_csv('causalai/tests/datasets/application/cloud_abnormal.csv')
        lower_level_columns = ['Customer DB', 'Shipping Cost Service', 'Caching Service', 'Product DB']
        upper_level_metric = df_normal['Product Service'].tolist() + df_abnormal['Product Service'].tolist()
        df_normal = df_normal[lower_level_columns]
        df_abnormal = df_abnormal[lower_level_columns]

        data_obj, var_names = rca_preprocess(
            data=[df_normal, df_abnormal],
            time_metric=upper_level_metric,
            time_metric_name='time'
        )

        model = RootCauseDetector(
            data_obj = data_obj,
            var_names=var_names,
            time_metric_name='time',
            prior_knowledge=None
            )
        
        root_causes, graph = model.run(
            pvalue_thres=0.001,
            max_condition_set_size=4,
            return_graph=True
        )
        self.assertTrue(root_causes=={'Caching Service'})

# if __name__ == "__main__":
#     unittest.main()