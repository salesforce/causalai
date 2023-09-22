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
        graph_gt = {'A': {'value_dict': {('A', 0): 0.0, ('B', 0): 0.0, ('C', 0): 0.0, ('D', 0): 0.0, ('A', -1): 0.6882416978331496, 
                                ('B', -1): -0.37166057572834554, ('C', -1): -0.005783076770178308, ('D', -1): 0.07295711616369818, 
                                ('A', -2): 0.009934934327776451, ('B', -2): -0.010934427151092857, ('C', -2): 0.009226200018117386, ('D', -2): 0.0052691497218540165}, 
                          'pvalue_dict': {('A', 0): 1.0, ('B', 0): 1.0, ('C', 0): 1.0, ('D', 0): 1.0, ('A', -1): 0.0, ('B', -1): 0.0, 
                                ('C', -1): 0.39054354893499454, ('D', -1): 1.3625969770345196e-148, ('A', -2): 0.36413694519534523, 
                                ('B', -2): 0.20854719386594536, ('C', -2): 0.13005189998296762, ('D', -2): 0.18043629986926346}, 
                          'parents': [('A', -1), ('B', -1), ('D', -1)]}, 
                    'B': {'value_dict': {('A', 0): 0.0, ('B', 0): 0.0, ('C', 0): 0.0, ('D', 0): 0.0, ('A', -1): 0.0050864953117173125, 
                                ('B', -1): 0.806707729175584, ('C', -1): 0.011827808620149594, ('D', -1): 0.32714635249586976, 
                                ('A', -2): -0.01837640189094142, ('B', -2): -0.004467309626099758, ('C', -2): -0.029486641247070955, ('D', -2): 0.00647543586385744}, 
                          'pvalue_dict': {('A', 0): 1.0, ('B', 0): 1.0, ('C', 0): 1.0, ('D', 0): 1.0, ('A', -1): 0.8642613764585503, 
                                ('B', -1): 0.0, ('C', -1): 0.4034334309899529, ('D', -1): 0.0, ('A', -2): 0.424503604085278, ('B', -2): 0.8068730939686722, 
                                ('C', -2): 0.021357947145989646, ('D', -2): 0.43351841569009053}, 'parents': [('B', -1), ('D', -1)]}, 
                    'C': {'value_dict': {('A', 0): 0.0, ('B', 0): 0.0, ('C', 0): 0.0, ('D', 0): 0.0, ('A', -1): 0.008121680137858868, 
                                ('B', -1): -0.0069330044867996765, ('C', -1): 0.49751140933949933, ('D', -1): 0.216481528063104, 
                                ('A', -2): 0.0014255803132450229, ('B', -2): 0.4491625414844092, ('C', -2): 0.011250946166206588, ('D', -2): -0.0011395470512686476}, 
                          'pvalue_dict': {('A', 0): 1.0, ('B', 0): 1.0, ('C', 0): 1.0, ('D', 0): 1.0, ('A', -1): 0.7609616791791499, 
                                ('B', -1): 0.5853643072409201, ('C', -1): 7.541418834202973e-293, ('D', -1): 0.0, ('A', -2): 0.9449481851325486, 
                                ('B', -2): 4.8177524145806054e-154, ('C', -2): 0.32759178849579906, ('D', -2): 0.8779156550734817}, 
                          'parents': [('C', -1), ('B', -2), ('D', -1)]}, 
                    'D': {'value_dict': {('A', 0): 0.0, ('B', 0): 0.0, ('C', 0): 0.0, ('D', 0): 0.0, ('A', -1): -0.03584143233926987, 
                                ('B', -1): -0.028440621836902047, ('C', -1): 0.04097317188633277, ('D', -1): 0.40174562793047014, 
                                ('A', -2): 0.024490812086115157, ('B', -2): -0.01179541104165011, ('C', -2): -0.02211737950951286, ('D', -2): 0.01678241431063761}, 
                          'pvalue_dict': {('A', 0): 1.0, ('B', 0): 1.0, ('C', 0): 1.0, ('D', 0): 1.0, ('A', -1): 0.6275652295300248, ('B', -1): 0.4186554183190536, 
                                ('C', -1): 0.24374843593515036, ('D', -1): 5.014922915307754e-164, ('A', -2): 0.6681478941558929, ('B', -2): 0.7948869194187187, 
                                ('C', -2): 0.48675165874970927, ('D', -2): 0.4136362378051136}, 'parents': [('D', -1)]}}

        for key in graph_gt.keys():
            self.assertTrue(results[key]==graph_gt[key])


# if __name__ == "__main__":
#     unittest.main()
