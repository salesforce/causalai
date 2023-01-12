import unittest
import numpy as np
import math
from causalai.data.tabular import TabularData

from causalai.data.data_generator import GenerateRandomTimeseriesSEM, GenerateRandomTabularSEM, DataGenerator


class TestDataGenerator(unittest.TestCase):
    def test_data_generator(self):
        sem = GenerateRandomTimeseriesSEM(var_names=['a', 'b', 'c', 'd', 'e'], max_num_parents=4, max_lag=4, seed=0)
        self.assertTrue(list(sem.keys())==['a', 'b', 'c', 'd', 'e'])
        self.assertTrue((sem['a'][0][0])==('e', -3))
        self.assertTrue((sem['a'][0][1])==0.1)
        self.assertTrue((sem['b'][0][0])==('b', -4))
        self.assertTrue((sem['b'][0][1])==0.1)
        
        
        data, var_names, graph = DataGenerator(sem, T=2, noise_fn=None,\
                                        intervention=None, discrete=False, nstates=10, seed=1)
        
        data_gt = [[ 1.62434536, -0.52817175,  0.86540763,  1.74481176,  0.3190391 ],
                  [-0.61175641, -1.07296862, -2.3015387,  -0.7612069,  -0.24937038]]
        graph_gt = {'a': [('e', -3)], 'b': [('b', -4), ('a', -4), ('c', -2)],\
                    'c': [('e', -3), ('c', -4)], 'd': [('c', -3), ('b', -2)],\
                    'e': [('d', -1), ('b', -3), ('e', -1), ('c', -1)]}
        self.assertTrue(np.allclose(data, data_gt, atol=1e-7))
        self.assertTrue((var_names==['a', 'b', 'c', 'd', 'e']))
        self.assertTrue((graph==graph_gt))
        
        
        
        sem = GenerateRandomTabularSEM(var_names=['a', 'b', 'c', 'd', 'e'], max_num_parents=4, seed=0)
        self.assertTrue(list(sem.keys())==['a', 'b', 'c', 'd', 'e'])
        self.assertTrue((sem['a'][0][0])=='e')
        self.assertTrue((sem['a'][0][1])==0.1)
        self.assertTrue((sem['b'][0][0])=='d')
        self.assertTrue((sem['b'][0][1])==0.1)
        
        
        
        data, var_names, graph = DataGenerator(sem, T=2, noise_fn=None,\
                                        intervention=None, discrete=False, nstates=10, seed=1)
        data_gt = [[ 1.65624927, -0.35050018,  1.07498311,  1.77671567,  0.3190391 ],
                     [-0.63669345, -1.15158302, -2.40509013, -0.78614394, -0.24937038]]
        graph_gt = {'a': ['e'], 'b': ['d'], 'c': ['d', 'e'], 'd': ['e'], 'e': []}
        self.assertTrue(np.allclose(data, data_gt, atol=1e-7))
        self.assertTrue((var_names==['a', 'b', 'c', 'd', 'e']))
        self.assertTrue((graph==graph_gt))
        
        
        
        data, var_names, graph = DataGenerator(sem, T=10, noise_fn=None,\
                                        intervention=None, discrete=True, nstates=10, seed=1)
        data_gt =   [[4, 9, 7, 7, 5],
                     [2, 1, 0, 6, 1],
                     [7, 3, 5, 3, 0],
                     [0, 2, 9, 0, 2],
                     [9, 4, 2, 1, 3],
                     [3, 6, 6, 8, 6],
                     [6, 0, 4, 4, 7],
                     [5, 8, 1, 2, 9],
                     [8, 7, 3, 5, 4],
                     [1, 5, 8, 9, 8]]
        graph_gt = {'a': ['e'], 'b': ['d'], 'c': ['d', 'e'], 'd': ['e'], 'e': []}
        self.assertTrue(np.allclose(data, data_gt, atol=1e-7))
        self.assertTrue((var_names==['a', 'b', 'c', 'd', 'e']))
        self.assertTrue((graph==graph_gt))
        
        
        

# if __name__ == "__main__":
#     unittest.main()