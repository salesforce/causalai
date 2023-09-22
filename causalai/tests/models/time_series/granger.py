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
from causalai.models.time_series.granger import GrangerSingle, Granger

class TestGrangerTimeSeriesModel(unittest.TestCase):
    def test_granger_time_series_model(self):
        with open(f'causalai/tests/datasets/time_series/synthetic_data1.pkl', 'rb') as f:
            dataset = pkl.load(f)
        graph_gt, data_array = dataset['graph'], dataset['data']
        var_names = list(graph_gt.keys())

        data_train = data_array

        StandardizeTransform_ = StandardizeTransform()
        StandardizeTransform_.fit(data_train)

        data_train_trans = StandardizeTransform_.transform(data_train)

        data_train_obj = TimeSeriesData(data_train_trans, var_names=var_names)
        prior_knowledge = PriorKnowledge(forbidden_links={'A': ['C']}) # C cannot be a parent of A

        model = GrangerSingle(data=data_train_obj, prior_knowledge=prior_knowledge, use_multiprocessing=False, max_iter=1000, cv=5)

        target_var = 'A'
        max_lag = 2

        # check if candidate parent are correctly extracted
        candidate_parents = model.get_candidate_parents(target_var, max_lag)
        gt = [('A', -1), ('A', -2), ('B', -1), ('B', -2), ('D', -1), ('D', -2)]
        self.assertTrue(candidate_parents, gt)

        # check if arrays extracted given variable names and max_lag are correct
        X = target_var
        Z = candidate_parents
        x,_,z = model.data.extract_array(X=X, Y=None, Z=Z, max_lag=max_lag)
        gt = [ 0.04214253,  0.16161745,  0.19469583,  1.11968867,  0.78705638,  0.35639302,
               0.12113487,  0.11813991, -0.23865448, -0.38674358]
        self.assertTrue(np.allclose(x[:10], gt, atol=1e-7))
        gt =[[-0.09560413,  0.01568265, -0.10188444, -0.09077149, -0.01187416],
             [ 0.04214253, -0.09560413, -0.60488271, -0.10188444,  0.07670529],
             [ 0.16161745,  0.04214253, -0.84314235, -0.60488271, -0.02387971],
             [ 0.19469583,  0.16161745, -1.24307748, -0.84314235,  1.66495893],
             [ 1.11968867,  0.19469583, -0.36360436, -1.24307748,  0.36948041]]
        self.assertTrue(np.allclose(z[:5,:5], gt, atol=1e-7))


        # check the outputs of GrangerSingle methods
        target_var = 'A'
        max_lag = 2
        granger_single = GrangerSingle(
            data=data_train_obj,
            prior_knowledge=prior_knowledge,
            max_iter=1000, # number of optimization iterations for model fitting (default value is 1000)
            use_multiprocessing=False
            )
        results = granger_single.run(target_var=target_var, max_lag=max_lag, pvalue_thres=0.05)
        gt = {('A', -1): 0.6946352328997523, ('A', -2): 0.0025173929468375327,\
              ('B', -1): -0.3682079561558271, ('B', -2): -0.009627641466785446,\
              ('C', -1): None, ('C', -2): None, ('D', -1): 0.07229391846228861,\
              ('D', -2): 0.0019945461610456345}
        for k,v in results['value_dict'].items():
            self.assertTrue(k in gt)
            if v is None:
                self.assertTrue(gt[k]==v)
            else:
                self.assertTrue(np.allclose(v, gt[k], atol=1e-7))

         
        gt = {('A', -1): 0.0, ('A', -2): 0.8102997834810995, ('B', -1): 0.0,\
              ('B', -2): 0.24832571023192063, ('C', -1): None, ('C', -2): None,\
              ('D', -1): 3.0677945470369538e-146, ('D', -2): 0.5901938258900048}
        for k,v in results['pvalue_dict'].items():
            self.assertTrue(k in gt)
            if v is None:
                self.assertTrue(gt[k]==v)
            else:
                self.assertTrue(np.allclose(v, gt[k], atol=1e-7))


        parents = granger_single.get_parents(pvalue_thres=0.05)
        gt = [('A', -1), ('B', -1), ('D', -1)]
        self.assertTrue(parents==gt)



        # check the outputs of Full Causal Discovery methods
        prior_knowledge = None
        max_lag = 2

        granger = Granger(
                data=data_train_obj,
                prior_knowledge=prior_knowledge,
                max_iter=1000, # number of optimization iterations for model fitting (default value is 1000)
                use_multiprocessing=False
                )

        results = granger.run(max_lag=max_lag, pvalue_thres=0.1)
#         print(results)
        graph_gt = {'A': {'parents': [('A', -1), ('B', -1), ('D', -1)], 
        'value_dict': {('A', -1): 0.6960504486081752, ('A', -2): 0.0014181763497011757, ('B', -1): -0.3671415428291086, ('B', -2): -0.009641970232916484, 
                                ('C', -1): -0.0, ('C', -2): 0.0, ('D', -1): 0.07206176055281632, ('D', -2): 0.0012596247064715575}, 
        'pvalue_dict': {('A', -1): 0.0, ('A', -2): 0.896921305480296, ('B', -1): 0.0, ('B', -2): 0.26744553442346214, ('C', -1): 1.0, ('C', -2): 1.0, 
                                ('D', -1): 2.9629987034884055e-145, ('D', -2): 0.7487951985973491}}, 
'B': {'parents': [('B', -1), ('D', -1)], 
        'value_dict': {('A', -1): -0.0, ('A', -2): -0.010483044174705832, ('B', -1): 0.8010887999130821, ('B', -2): -0.0, ('C', -1): 0.00397121116329935, 
                                ('C', -2): -0.019455828150947853, ('D', -1): 0.3265395990339475, ('D', -2): 0.008802295547999106}, 
        'pvalue_dict': {('A', -1): 1.0, ('A', -2): 0.6485782632746502, ('B', -1): 0.0, ('B', -2): 1.0, ('C', -1): 0.7790044486215731, 
                                ('C', -2): 0.12868051554141427, ('D', -1): 0.0, ('D', -2): 0.2869174022399909}}, 
'C': {'parents': [('C', -1), ('B', -2), ('D', -1)], 
        'value_dict': {('A', -1): -0.0, ('A', -2): -0.0, ('B', -1): 0.0, ('B', -2): 0.438727913930932, ('C', -1): 0.494264059001986, ('C', -2): 0.006620230210262063, 
                                ('D', -1): 0.21319167123510155, ('D', -2): 0.0}, 
        'pvalue_dict': {('A', -1): 1.0, ('A', -2): 1.0, ('B', -1): 1.0, ('B', -2): 1.4173129941263553e-147, ('C', -1): 1.1631811205136963e-289, 
                                ('C', -2): 0.5644908686711352, ('D', -1): 0.0, ('D', -2): 1.0}}, 
'D': {'parents': [('D', -1)], 
        'value_dict': {('A', -1): 0.0, ('A', -2): -0.0, ('B', -1): -0.0, ('B', -2): -0.0, ('C', -1): 0.0, ('C', -2): 0.0, ('D', -1): 0.3881938025602495, ('D', -2): 0.0}, 
        'pvalue_dict': {('A', -1): 1.0, ('A', -2): 1.0, ('B', -1): 1.0, ('B', -2): 1.0, ('C', -1): 1.0, ('C', -2): 1.0, ('D', -1): 5.589775953001244e-154, ('D', -2): 1.0}}}

        for key in graph_gt.keys():
            self.assertTrue(results[key]==graph_gt[key])


# if __name__ == "__main__":
#     unittest.main()





