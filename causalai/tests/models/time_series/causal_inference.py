import unittest
import numpy as np
import math
import pickle as pkl


from causalai.data.data_generator import DataGenerator, ConditionalDataGenerator
from causalai.models.time_series.causal_inference import CausalInference
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from functools import partial

def define_treatments(name, t,c):
    treatment = dict(var_name=name,
                    treatment_value=t,
                    control_value=c)
    return treatment

class TestCausalInferenceTimeSeriesModel(unittest.TestCase):
    def test_ate_timeseries_model(self):
        fn = lambda x:x
        coef = 1.5
        sem = {
                'a': [], 
                'b': [(('a', -1), coef, fn), (('f', -1), coef, fn)], 
                'c': [(('b', -2), coef, fn), (('f', -2), coef, fn)],
                'd': [(('b', -4), coef, fn), (('b', -1), coef, fn), (('g', -1), coef, fn)],
                'e': [(('f', -1), coef, fn)], 
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
        self.assertTrue(f'{ate:.2f}'=='18.02')
    
    def test_cate_timeseries_model(self):
        T=5000
        data, var_names, graph_gt = ConditionalDataGenerator(T=T, data_type='time_series', seed=0, discrete=False)
        # var_names = ['C', 'W', 'X', 'Y']
        treatment_var='X'
        target = 'Y'
        target_idx = var_names.index(target)

        intervention1 = 0.1*np.ones(T, dtype=float)
        intervention_data1,_,_ = ConditionalDataGenerator(T=T, data_type='time_series',\
                                            seed=0, intervention={treatment_var:intervention1}, discrete=False)

        intervention2 = 0.9*np.ones(T, dtype=float)
        intervention_data2,_,_ = ConditionalDataGenerator(T=T, data_type='time_series',\
                                            seed=0, intervention={treatment_var:intervention2}, discrete=False)

        condition_state=2.1
        diff = np.abs(data[:,0] - condition_state)
        idx = np.argmin(diff)

        cate_gt = (intervention_data1[idx,target_idx] - intervention_data2[idx,target_idx])

        treatments = define_treatments(treatment_var, intervention1,intervention2)
        conditions = {'var_name': 'C', 'condition_value': condition_state}

        np.random.seed(0)
        model = partial(MLPRegressor, hidden_layer_sizes=(100,100), max_iter=200)
        CausalInference_ = CausalInference(data, var_names, graph_gt, model, discrete=False)#

        cate = CausalInference_.cate(target, treatments, conditions, model)
        
        self.assertTrue(f'{cate:.2f}'=='-1.63')
        self.assertTrue(f'{cate_gt:.2f}'=='-1.69')
    
    def test_counterfactual_timeseries_model(self):
        fn = lambda x:x
        coef = 0.1
        sem = {
                'a': [], 
                'b': [(('a', -1), coef, fn), (('f', -1), coef, fn)], 
                'c': [(('b', 0), coef, fn), (('f', -2), coef, fn)],
                'd': [(('b', -4), coef, fn), (('g', -1), coef, fn)],
                'e': [(('f', -1), coef, fn)], 
                'f': [],
                'g': [],
                }
        T = 5000
        data,var_names,graph_gt = DataGenerator(sem, T=T, seed=0)

        intervention={'b':np.array([10.]*10), 'e':np.array([-100.]*10)}
        target_var = 'c'

        sample, _, _= DataGenerator(sem, T=10, noise_fn=None,\
                                            intervention=None, discrete=False, nstates=10, seed=0)
        sample_intervened, _, _= DataGenerator(sem, T=10, noise_fn=None,\
                                            intervention=intervention, discrete=False, nstates=10, seed=0)

        sample=sample[-1] # use the last time step as our sample
        sample_intervened=sample_intervened[-1] # use the last time step as our sample and compute ground truth intervention
        var_orig = sample[var_names.index(target_var)]
        var_counterfactual_gt = sample_intervened[var_names.index(target_var)] # ground truth counterfactual


        interventions = {name:float(val[0]) for name, val in intervention.items()}

        model = LinearRegression
        CausalInference_ = CausalInference(data, var_names, graph_gt, model, discrete=False)
        counterfactual_et = CausalInference_.counterfactual(sample, target_var, interventions, model)
        self.assertTrue(f'{counterfactual_et:.2f}'=='1.26')
        self.assertTrue(f'{var_counterfactual_gt:.2f}'=='1.16')


# test = TestCausalInferenceTimeSeriesModel()
# test.test_ate_timeseries_model()
# test.test_cate_timeseries_model()
# test.test_counterfactual_timeseries_model()

# if __name__ == "__main__":
#     unittest.main()
