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
from causalai.models.time_series.pc import PCSingle, PC

class TestPCTimeSeriesModel(unittest.TestCase):
    def test_pc_time_series_model(self):
        with open('causalai/tests/datasets/time_series/synthetic_data1.pkl', 'rb') as f:
            dataset = pkl.load(f)
        graph_gt, data_array = dataset['graph'], dataset['data']
        var_names = list(graph_gt.keys())

        data_train = data_array

        StandardizeTransform_ = StandardizeTransform()
        StandardizeTransform_.fit(data_train)

        data_train_trans = StandardizeTransform_.transform(data_train)

        data_train_obj = TimeSeriesData(data_train_trans, var_names=var_names)
        prior_knowledge = PriorKnowledge(forbidden_links={'A': ['C']}) # C cannot be a parent of A

        model = PCSingle(data=data_train_obj, prior_knowledge=prior_knowledge, use_multiprocessing=False)

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
        pc_single = PCSingle(
            data=data_train_obj,
            prior_knowledge=prior_knowledge,
            use_multiprocessing=False
            )
        results = pc_single.run(target_var=target_var, max_lag=max_lag, pvalue_thres=0.05)
        gt = {('A', -1): 0.5671264368932202, ('B', -1): -0.615603416125299, ('D', -2): 0.018967855018957838}
        for k,v in results['value_dict'].items():
            self.assertTrue(k in gt)
            if v is None:
                self.assertTrue(gt[k]==v)
            else:
                self.assertTrue(np.allclose(v, gt[k], atol=1e-7))

         
        gt = {('A', -1): 0.0, ('B', -1): 0.0, ('D', -2): 0.18030806177323247}
        for k,v in results['pvalue_dict'].items():
            self.assertTrue(k in gt)
            if v is None:
                self.assertTrue(gt[k]==v)
            else:
                self.assertTrue(np.allclose(v, gt[k], atol=1e-7))


        parents = pc_single.get_parents(pvalue_thres=0.05)
        gt = [('B', -1), ('A', -1)]
        self.assertTrue(parents==gt)



        # check the outputs of Full Causal Discovery methods
        prior_knowledge = None
        max_lag = 2

        pc = PC(
                data=data_train_obj,
                prior_knowledge=prior_knowledge,
                use_multiprocessing=False
                )

        results = pc.run(max_lag=max_lag, pvalue_thres=0.1)
        graph_gt = {'A': {'parents': [('B', -1), ('A', -1), ('D', -1)],\
                    'value_dict': {('A', -1): 0.5671264368932202, ('A', -2): 0.012852350470859421,\
                    ('B', -1): -0.615603416125299, ('B', -2): -0.017809295837559393,\
                    ('C', -1): -0.012159984152341258, ('C', -2): 0.021438269934095255,\
                    ('D', -1): 0.3555952907346426, ('D', -2): 0.018967855018957838},\
                    'pvalue_dict': {('A', -1): 0.0, ('A', -2): 0.3639890823495211,\
                    ('B', -1): 0.0, ('B', -2): 0.20840690648157942, ('C', -1): 0.39040501803127814,\
                    ('C', -2): 0.1299385314261815, ('D', -1): 1.1121434892885163e-148,\
                    ('D', -2): 0.18030806177323247}}, 'B': {'parents': [('D', -1), ('B', -1), ('C', -2)],\
                    'value_dict': {('A', -1): 0.0024213815361161547, ('A', -2): -0.01131063212244077,\
                    ('B', -1): 0.6278672248094768, ('B', -2): -0.0034619252300099137,\
                    ('C', -1): 0.011832356473849451, ('C', -2): -0.032587874077170925,\
                    ('D', -1): 0.6302064401432064, ('D', -2): 0.011091931710231312},\
                    'pvalue_dict': {('A', -1): 0.8642074356586027, ('A', -2): 0.4243555073659325,\
                    ('B', -1): 0.0, ('B', -2): 0.8068326541024651, ('C', -1): 0.4033009189407363,\
                    ('C', -2): 0.021319914898052638, ('D', -1): 0.0, ('D', -2): 0.43336899299613396}},\
                    'C': {'parents': [('D', -1), ('C', -1), ('B', -2)],\
                    'value_dict': {('A', -1): 0.00430884293720442, ('A', -2): 0.0009776854441377934,\
                    ('B', -1): -0.007726849396160212, ('B', -2): 0.3617039237881151,\
                    ('C', -1): 0.4851010218471578, ('C', -2): 0.013864600487085772,\
                    ('D', -1): 0.5136197490669024, ('D', -2): -0.0021752088631017425},\
                    'pvalue_dict': {('A', -1): 0.760874942040977, ('A', -2): 0.9449472181793205,\
                    ('B', -1): 0.5852376459032218, ('B', -2): 3.899998165119913e-154,\
                    ('C', -1): 5.043535187772342e-293, ('C', -2): 0.32743459131550623,\
                    ('D', -1): 0.0, ('D', -2): 0.8778985699450377}}, 'D': {'parents': [('D', -1)],\
                    'value_dict': {('A', -1): -0.0068709968797109255, ('A', -2): 0.0060713264310978034,\
                    ('B', -1): -0.011453821670307127, ('B', -2): -0.0036818072768950794,\
                    ('C', -1): 0.016507833749922755, ('C', -2): -0.00984964500008735,\
                    ('D', -1): 0.37255378365043496, ('D', -2): 0.011577815247833707},\
                    'pvalue_dict': {('A', -1): 0.627463892023082, ('A', -2): 0.6680559885895293,\
                    ('B', -1): 0.41851393586777286, ('B', -2): 0.7948295218008653,\
                    ('C', -1): 0.24360780602533996, ('C', -2): 0.4866226477317521,\
                    ('D', -1): 4.006749398276717e-164, ('D', -2): 0.4134939244597493}}}

        for key in graph_gt.keys():
            self.assertTrue(results[key]==graph_gt[key])


# if __name__ == "__main__":
#     unittest.main()


