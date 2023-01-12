import unittest
import numpy as np
import math
import pickle as pkl


from causalai.data.data_generator import DataGenerator
from causalai.models.tabular.causal_inference import CausalInference
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

def define_treatments(name, t,c):
    treatment = dict(var_name=name,
                    treatment_value=t,
                    control_value=c)
    return treatment

class TestCausalInferenceTabularModel(unittest.TestCase):
    def test_causal_inference_tabular_model(self):
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
        CausalInference_ = CausalInference(data_array, var_names, graph_gt, LinearRegression , discrete=False)

        ate, y_treat,y_control = CausalInference_.ate(target, treatments)

        self.assertTrue(f'{ate:.2f}'=='17.98')


# if __name__ == "__main__":
#     unittest.main()
