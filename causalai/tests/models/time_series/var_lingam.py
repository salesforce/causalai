import unittest
import numpy as np
import math
import pickle as pkl
try:
    import ray
except:
    pass

from causalai.models.common.prior_knowledge import PriorKnowledge
from causalai.data.time_series import TimeSeriesData
from causalai.data.transforms.time_series import StandardizeTransform
from causalai.models.common.prior_knowledge import PriorKnowledge
from causalai.models.time_series.var_lingam import VARLINGAM

class TestVARLINGAMTimeSeriesModel(unittest.TestCase):
    def test_varlingam_time_series_model(self):
        with open('causalai/tests/datasets/time_series/synthetic_data1.pkl', 'rb') as f:
            dataset = pkl.load(f)
        graph_gt, data_array = dataset['graph'], dataset['data']
        var_names = list(graph_gt.keys())

        data_train = data_array

        StandardizeTransform_ = StandardizeTransform()
        StandardizeTransform_.fit(data_train)

        data_train_trans = StandardizeTransform_.transform(data_train)

        data_train_obj = TimeSeriesData(data_train_trans, var_names=var_names)

        max_lag = 2

        varlingam = VARLINGAM(data=data_train_obj)

        results = varlingam.run(max_lag=max_lag, pvalue_thres=0.01)
#         print(results)
        graph_gt = {'A': {'value_dict': {('A', 0): 0.0, ('B', 0): 0.0, ('C', 0): 0.0,\
                    ('D', 0): 0.0, ('A', -1): 0.6882416978331495, ('B', -1): -0.3716605757283459,\
                    ('C', -1): -0.005783076770177935, ('D', -1): 0.0729571161636981,\
                    ('A', -2): 0.00993493432777656, ('B', -2): -0.010934427151092723,\
                    ('C', -2): 0.009226200018117586, ('D', -2): 0.0052691497218540296},\
                    'pvalue_dict': {('A', 0): 1.0, ('B', 0): 1.0, ('C', 0): 1.0, ('D', 0): 1.0,\
                    ('A', -1): 0.0, ('B', -1): 0.0, ('C', -1): 0.39054354893502474,\
                    ('D', -1): 1.3625969770356058e-148, ('A', -2): 0.3641369451953359,\
                    ('B', -2): 0.20854719386595116, ('C', -2): 0.13005189998295943,\
                    ('D', -2): 0.1804362998692613}, 'parents': [('A', -1), ('B', -1), ('D', -1)]},\
                    'B': {'value_dict': {('A', 0): 0.0, ('B', 0): 0.0, ('C', 0): 0.0,\
                    ('D', 0): 0.0, ('A', -1): 0.005086495311717896, ('B', -1): 0.8067077291755844,\
                    ('C', -1): 0.011827808620149064, ('D', -1): 0.32714635249586993,\
                    ('A', -2): -0.01837640189094182, ('B', -2): -0.00446730962609971,\
                    ('C', -2): -0.029486641247071045, ('D', -2): 0.006475435863856788},\
                    'pvalue_dict': {('A', 0): 1.0, ('B', 0): 1.0, ('C', 0): 1.0, ('D', 0): 1.0,\
                    ('A', -1): 0.8642613764585338, ('B', -1): 0.0, ('C', -1): 0.4034334309899743,\
                    ('D', -1): 0.0, ('A', -2): 0.424503604085264, ('B', -2): 0.8068730939686742,\
                    ('C', -2): 0.02135794714598931, ('D', -2): 0.4335184156901358},\
                    'parents': [('B', -1), ('D', -1)]}, 'C': {'value_dict': {('A', 0): 0.0,\
                    ('B', 0): 0.0, ('C', 0): 0.0, ('D', 0): 0.0, ('A', -1): 0.008121680137859024,\
                    ('B', -1): -0.006933004486799621, ('C', -1): 0.49751140933949956,\
                    ('D', -1): 0.21648152806310392, ('A', -2): 0.0014255803132450198,\
                    ('B', -2): 0.44916254148440915, ('C', -2): 0.01125094616620633,\
                    ('D', -2): -0.0011395470512688374}, 'pvalue_dict': {('A', 0): 1.0, ('B', 0): 1.0,\
                    ('C', 0): 1.0, ('D', 0): 1.0, ('A', -1): 0.7609616791791436,\
                    ('B', -1): 0.5853643072409223, ('C', -1): 7.541418834199538e-293,\
                    ('D', -1): 0.0, ('A', -2): 0.9449481851325483, ('B', -2): 4.817752414581976e-154,\
                    ('C', -2): 0.32759178849581083, ('D', -2): 0.8779156550734611},\
                    'parents': [('C', -1), ('B', -2), ('D', -1)]}, 'D': {'value_dict': {('A', 0): 0.0,\
                    ('B', 0): 0.0, ('C', 0): 0.0, ('D', 0): 0.0, ('A', -1): -0.035841432339269584,\
                    ('B', -1): -0.028440621836902234, ('C', -1): 0.0409731718863327,\
                    ('D', -1): 0.40174562793047036, ('A', -2): 0.024490812086114637,\
                    ('B', -2): -0.01179541104164999, ('C', -2): -0.02211737950951284,\
                    ('D', -2): 0.016782414310637258}, 'pvalue_dict': {('A', 0): 1.0,\
                    ('B', 0): 1.0, ('C', 0): 1.0, ('D', 0): 1.0, ('A', -1): 0.6275652295300248,\
                    ('B', -1): 0.4186554183190495, ('C', -1): 0.243748435935151,\
                    ('D', -1): 5.014922915305755e-164, ('A', -2): 0.6681478941558969,\
                    ('B', -2): 0.7948869194187206, ('C', -2): 0.4867516587497096,\
                    ('D', -2): 0.4136362378051224}, 'parents': [('D', -1)]}}

        for key in graph_gt.keys():
            self.assertTrue(results[key]==graph_gt[key])


# if __name__ == "__main__":
#     unittest.main()
