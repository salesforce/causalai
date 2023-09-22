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
        graph_gt = {'A': {'parents': [('B', -1), ('A', -1), ('D', -1)], 
                         'value_dict': {('A', -1): 0.5671264368932201, ('A', -2): 0.012852350470859418, ('B', -1): -0.615603416125299, 
                                    ('B', -2): -0.01780929583755938, ('C', -1): -0.012159984152341265, ('C', -2): 0.02143826993409525, 
                                    ('D', -1): 0.35559529073464263, ('D', -2): 0.018967855018957834}, 
                         'pvalue_dict': {('A', -1): 0.0, ('A', -2): 0.36398908234952165, ('B', -1): 0.0, ('B', -2): 0.20840690648157983, 
                                    ('C', -1): 0.3904050180312776, ('C', -2): 0.1299385314261816, ('D', -1): 1.1121434892885163e-148, ('D', -2): 0.18030806177323255}}, 
                    'B': {'parents': [('D', -1), ('B', -1), ('C', -2)], 
                          'value_dict': {('A', -1): 0.002421381536116151, ('A', -2): -0.011310632122440795, ('B', -1): 0.6278672248094768, 
                                    ('B', -2): -0.003461925230009909, ('C', -1): 0.011832356473849441, ('C', -2): -0.032587874077170904, 
                                    ('D', -1): 0.6302064401432064, ('D', -2): 0.011091931710231309}, 
                          'pvalue_dict': {('A', -1): 0.8642074356586029, ('A', -2): 0.42435550736593164, ('B', -1): 0.0, ('B', -2): 0.8068326541024654, 
                                    ('C', -1): 0.4033009189407367, ('C', -2): 0.021319914898052704, ('D', -1): 0.0, ('D', -2): 0.43336899299613396}}, 
                    'C': {'parents': [('D', -1), ('C', -1), ('B', -2)], 
                          'value_dict': {('A', -1): 0.004308842937204402, ('A', -2): 0.000977685444137783, ('B', -1): -0.007726849396160209, 
                                    ('B', -2): 0.36170392378811506, ('C', -1): 0.4851010218471578, ('C', -2): 0.013864600487085774, 
                                    ('D', -1): 0.5136197490669024, ('D', -2): -0.0021752088631017273}, 
                          'pvalue_dict': {('A', -1): 0.760874942040978, ('A', -2): 0.944947218179321, ('B', -1): 0.5852376459032221, 
                                    ('B', -2): 3.8999981651210235e-154, ('C', -1): 5.043535187772342e-293, ('C', -2): 0.32743459131550623, 
                                    ('D', -1): 0.0, ('D', -2): 0.8778985699450386}}, 'D': {'parents': [('D', -1)], 
                          'value_dict': {('A', -1): -0.006870996879710911, ('A', -2): 0.006071326431097805, ('B', -1): -0.011453821670307136, 
                                    ('B', -2): -0.0036818072768950694, ('C', -1): 0.016507833749922738, ('C', -2): -0.00984964500008734, 
                                    ('D', -1): 0.372553783650435, ('D', -2): 0.011577815247833717}, 
                          'pvalue_dict': {('A', -1): 0.6274638920230825, ('A', -2): 0.6680559885895293, ('B', -1): 0.41851393586777297, 
                                    ('B', -2): 0.7948295218008659, ('C', -1): 0.24360780602534055, ('C', -2): 0.48662264773175246, 
                                    ('D', -1): 4.0067493982767165e-164, ('D', -2): 0.41349392445974886}}}

        for key in graph_gt.keys():
            self.assertTrue(results[key]==graph_gt[key])


# if __name__ == "__main__":
#     unittest.main()


