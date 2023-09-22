
import unittest
import numpy as np
import math
import pickle as pkl


from causalai.data.data_generator import DataGenerator, ConditionalDataGenerator
from causalai.models.tabular.causal_inference import CausalInference
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from functools import partial

def define_treatments(name, t,c):
    treatment = dict(var_name=name,
                    treatment_value=t,
                    control_value=c)
    return treatment

class TestCausalInferenceTabularModel(unittest.TestCase):
    def test_ate_tabular_model(self):
        fn = lambda x:x
        coef = 1.5
        sem = {
                'a': [], 
                'b': [('a', coef, fn), ('f', coef, fn)], 
                'c': [('b', coef, fn), ('f', coef, fn)],
                'd': [('b', coef, fn), ('g', coef, fn)],
                'e': [('f', coef, fn)], 
                'f': [],
                'g': [],
                }
        T = 2000
        data_array, var_names, graph_gt = DataGenerator(sem, T=T, seed=0, discrete=False)


        t1='a' 
        t2='b'
        target = 'c'
        target_var = var_names.index(target)

        intervention11 = 1*np.ones(T)
        intervention21 = 10*np.ones(T)
        intervention_data1,_,_ = DataGenerator(sem, T=T, seed=7,
                                intervention={t1:intervention11, t2:intervention21})

        intervention12 = -0.*np.ones(T)
        intervention22 = -2.*np.ones(T)
        intervention_data2,_,_ = DataGenerator(sem, T=T, seed=7,
                                intervention={t1:intervention12, t2:intervention22})

        treatments = [define_treatments(t1, intervention11,intervention12),\
                     define_treatments(t2, intervention21,intervention22)]
        
        np.random.seed(0)
        CausalInference_ = CausalInference(data_array, var_names, graph_gt, LinearRegression , discrete=False)

        ate, y_treat,y_control = CausalInference_.ate(target, treatments)

        self.assertTrue(f'{ate:.2f}'=='17.98')
    
    def test_cate_tabular_model(self):
        T=500
        data, var_names, graph_gt = ConditionalDataGenerator(T=T, data_type='tabular', seed=0, discrete=False)
        # var_names = ['C', 'W', 'X', 'Y']
        treatment_var='X'
        target = 'Y'
        target_idx = var_names.index(target)


        intervention1 = 0.1*np.ones(T, dtype=int)
        intervention_data1,_,_ = ConditionalDataGenerator(T=T, data_type='tabular',\
                                            seed=0, intervention={treatment_var:intervention1}, discrete=False)

        intervention2 = 0.9*np.ones(T, dtype=int)
        intervention_data2,_,_ = ConditionalDataGenerator(T=T, data_type='tabular',\
                                            seed=0, intervention={treatment_var:intervention2}, discrete=False)
        
        
        condition_state=2.1
        diff = np.abs(data[:,0] - condition_state)
        idx = np.argmin(diff)

        cate_gt = (intervention_data1[idx,target_idx] - intervention_data2[idx,target_idx])

        ####
        treatments = define_treatments(treatment_var, intervention1,intervention2)
        conditions = {'var_name': 'C', 'condition_value': condition_state}

        np.random.seed(0)
        model = partial(MLPRegressor, hidden_layer_sizes=(100,100), max_iter=200)
        CausalInference_ = CausalInference(data, var_names, graph_gt, model, discrete=False, method='causal_path')#

        cate_cp = CausalInference_.cate(target, treatments, conditions, model)

        np.random.seed(0)
        model = partial(MLPRegressor, hidden_layer_sizes=(100,100), max_iter=200)
        CausalInference_ = CausalInference(data, var_names, graph_gt, model, discrete=False, method='backdoor')#

        cate_bd = CausalInference_.cate(target, treatments, conditions, model)
#         print(cate_bd,cate_gt, cate_cp)
        self.assertTrue(f'{cate_bd:.2f}'=='-1.75')
        self.assertTrue(f'{cate_gt:.2f}'=='-1.65')
        self.assertTrue(f'{cate_cp:.2f}'=='-1.77')

    def test_counterfactual_tabular_model(self):
        fn = lambda x:x
        coef = 0.5
        sem = {
                'a': [], 
                'b': [('a', coef, fn), ('f', coef, fn)], 
                'c': [('b', coef, fn), ('f', coef, fn)],
                'd': [('b', coef, fn), ('g', coef, fn)],
                'e': [('f', coef, fn)], 
                'f': [],
                'g': [],
                }
        T = 400
        data, var_names, graph_gt = DataGenerator(sem, T=T, seed=0, discrete=False)

        intervention={'a':np.array([10.]*10), 'e':np.array([-0.]*10)}
        target_var = 'c'

        sample, _, _= DataGenerator(sem, T=10, noise_fn=None,\
                                            intervention=None, discrete=False, nstates=10, seed=1)
        sample_intervened, _, _= DataGenerator(sem, T=10, noise_fn=None,\
                                            intervention=intervention, discrete=False, nstates=10, seed=1)

        sample=sample[0]
        sample_intervened=sample_intervened[0]
        var_counterfactual_gt = sample_intervened[var_names.index(target_var)]
        
        interventions = {name:float(val[0]) for name, val in intervention.items()}

        np.random.seed(0)
        model = LinearRegression

        CausalInference_ = CausalInference(data, var_names, graph_gt, model, discrete=False, method='causal_path')
        counterfactual_cp = CausalInference_.counterfactual(sample, target_var, interventions, model)

        np.random.seed(0)
        CausalInference_ = CausalInference(data, var_names, graph_gt, model, discrete=False, method='backdoor')
        counterfactual_bd = CausalInference_.counterfactual(sample, target_var, interventions, model)
    
        self.assertTrue(f'{var_counterfactual_gt:.2f}'=='3.33')
        self.assertTrue(f'{counterfactual_cp:.2f}'=='3.13')
        self.assertTrue(f'{counterfactual_bd:.2f}'=='1.45')
        
        
# test = TestCausalInferenceTabularModel()
# test.test_ate_tabular_model()
# test.test_cate_tabular_model()
# test.test_counterfactual_tabular_model()
