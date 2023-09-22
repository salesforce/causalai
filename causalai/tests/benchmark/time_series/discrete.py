
import unittest
import math
from causalai.benchmark.time_series.discrete import BenchmarkDiscreteTimeSeries

from functools import partial
import numpy as np

def define_treatments(name, t,c):
    treatment = dict(var_name=name,
                    treatment_value=t,
                    control_value=c)
    return treatment

class TestBenchmarkDiscreteTimeSeries(unittest.TestCase):
    def test_benchmark_graph_density(self):
        
        np.random.seed(0)
        b = BenchmarkDiscreteTimeSeries(algo_dict=None, kargs_dict=None, 
                                     num_exp=2, custom_metric_dict=None)
        b.benchmark_graph_density(graph_density_list = [0.1, 0.5], num_vars=5, T=500,\
                           fn= lambda x:x, coef=0.1, noise_fn=np.random.rand, data_max_lag=2)

        gt=[[{'PC-Pearson': {'time_taken': 0.6622037887573242,
            'precision': 0.6,
            'recall': 0.8,
            'f1_score': 0.6},
           'PC-Log-Likelihood': {'time_taken': 0.6444821357727051,
            'precision': 0.6,
            'recall': 0.8,
            'f1_score': 0.6},
           'PC-Mod-Log-Likelihood': {'time_taken': 0.6982431411743164,
            'precision': 0.6,
            'recall': 0.8,
            'f1_score': 0.6},
           'PC-Freeman-Tukey': {'time_taken': 0.6307289600372314,
            'precision': 0.6,
            'recall': 0.8,
            'f1_score': 0.6},
           'PC-Neyman': {'time_taken': 0.03506898880004883,
            'precision': 0.6,
            'recall': 0.8,
            'f1_score': 0.6}},
          {'PC-Pearson': {'time_taken': 0.10736298561096191,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'PC-Log-Likelihood': {'time_taken': 0.10596394538879395,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'PC-Mod-Log-Likelihood': {'time_taken': 0.6138238906860352,
            'precision': 0.8,
            'recall': 1.0,
            'f1_score': 0.8},
           'PC-Freeman-Tukey': {'time_taken': 0.11305904388427734,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'PC-Neyman': {'time_taken': 0.0378880500793457,
            'precision': 0.8,
            'recall': 1.0,
            'f1_score': 0.8}}],
         [{'PC-Pearson': {'time_taken': 0.6461501121520996,
            'precision': 0.0,
            'recall': 0.2,
            'f1_score': 0.0},
           'PC-Log-Likelihood': {'time_taken': 0.6259210109710693,
            'precision': 0.0,
            'recall': 0.2,
            'f1_score': 0.0},
           'PC-Mod-Log-Likelihood': {'time_taken': 0.628643274307251,
            'precision': 0.0,
            'recall': 0.2,
            'f1_score': 0.0},
           'PC-Freeman-Tukey': {'time_taken': 0.6145431995391846,
            'precision': 0.0,
            'recall': 0.2,
            'f1_score': 0.0},
           'PC-Neyman': {'time_taken': 0.03499579429626465,
            'precision': 0.0,
            'recall': 0.2,
            'f1_score': 0.0}},
          {'PC-Pearson': {'time_taken': 0.11155891418457031,
            'precision': 0.4,
            'recall': 0.4,
            'f1_score': 0.4},
           'PC-Log-Likelihood': {'time_taken': 0.11295795440673828,
            'precision': 0.4,
            'recall': 0.4,
            'f1_score': 0.4},
           'PC-Mod-Log-Likelihood': {'time_taken': 0.11020183563232422,
            'precision': 0.4,
            'recall': 0.4,
            'f1_score': 0.4},
           'PC-Freeman-Tukey': {'time_taken': 0.1103670597076416,
            'precision': 0.4,
            'recall': 0.4,
            'f1_score': 0.4},
           'PC-Neyman': {'time_taken': 0.03621101379394531,
            'precision': 0.4,
            'recall': 0.4,
            'f1_score': 0.4}}]]
    
        for l1,l2 in zip(gt, b.results_full):
            for g1,g2 in zip(l1, l2):
                for algo in g1.keys():
                    for metric in ['precision', 'recall', 'f1_score']:
                        self.assertTrue(f'{g1[algo][metric]:.2f}'==f'{g2[algo][metric]:.2f}')
    
    def test_benchmark_variable_complexity(self):
        
        np.random.seed(0)
        b = BenchmarkDiscreteTimeSeries(algo_dict=None, kargs_dict=None, 
                                     num_exp=2, custom_metric_dict=None)
        b.benchmark_variable_complexity(num_vars_list = [2,10], graph_density=0.1, T=500,\
                           fn = lambda x:x, coef=0.1, noise_fn=np.random.randn, data_max_lag=2)

        gt=[[{'PC-Pearson': {'time_taken': 0.04582405090332031,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'PC-Log-Likelihood': {'time_taken': 0.030038833618164062,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'PC-Mod-Log-Likelihood': {'time_taken': 0.5327277183532715,
            'precision': 0.5,
            'recall': 1.0,
            'f1_score': 0.5},
           'PC-Freeman-Tukey': {'time_taken': 0.5580832958221436,
            'precision': 0.5,
            'recall': 1.0,
            'f1_score': 0.5},
           'PC-Neyman': {'time_taken': 0.006058931350708008,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0}},
          {'PC-Pearson': {'time_taken': 0.021531105041503906,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'PC-Log-Likelihood': {'time_taken': 0.023688077926635742,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'PC-Mod-Log-Likelihood': {'time_taken': 0.026500225067138672,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'PC-Freeman-Tukey': {'time_taken': 0.023321866989135742,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'PC-Neyman': {'time_taken': 0.007781982421875,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0}}],
         [{'PC-Pearson': {'time_taken': 1.183938980102539,
            'precision': 0.5,
            'recall': 0.6,
            'f1_score': 0.5},
           'PC-Log-Likelihood': {'time_taken': 0.9884109497070312,
            'precision': 0.6999999900000009,
            'recall': 0.6999999900000009,
            'f1_score': 0.6999999850000013},
           'PC-Mod-Log-Likelihood': {'time_taken': 2.6606998443603516,
            'precision': 0.49999999000000095,
            'recall': 0.6999999900000009,
            'f1_score': 0.49999998500000126},
           'PC-Freeman-Tukey': {'time_taken': 2.632638931274414,
            'precision': 0.49999999000000095,
            'recall': 0.6999999900000009,
            'f1_score': 0.49999998500000126},
           'PC-Neyman': {'time_taken': 1.7021009922027588,
            'precision': 0.49999999000000095,
            'recall': 0.6999999900000009,
            'f1_score': 0.49999998500000126}},
          {'PC-Pearson': {'time_taken': 2.161144256591797,
            'precision': 0.5,
            'recall': 0.8,
            'f1_score': 0.5},
           'PC-Log-Likelihood': {'time_taken': 2.077500104904175,
            'precision': 0.6,
            'recall': 0.8,
            'f1_score': 0.6},
           'PC-Mod-Log-Likelihood': {'time_taken': 3.983081102371216,
            'precision': 0.5,
            'recall': 0.8,
            'f1_score': 0.5},
           'PC-Freeman-Tukey': {'time_taken': 3.8389949798583984,
            'precision': 0.5,
            'recall': 0.8,
            'f1_score': 0.5},
           'PC-Neyman': {'time_taken': 3.536588191986084,
            'precision': 0.4,
            'recall': 0.8,
            'f1_score': 0.4}}]]
        
        for l1,l2 in zip(gt, b.results_full):
            for g1,g2 in zip(l1, l2):
                for algo in g1.keys():
                    for metric in ['precision', 'recall', 'f1_score']:
                        self.assertTrue(f'{g1[algo][metric]:.2f}'==f'{g2[algo][metric]:.2f}')
    
    def test_benchmark_sample_complexity(self):
        
        np.random.seed(0)
        b = BenchmarkDiscreteTimeSeries(algo_dict=None, kargs_dict=None, 
                                     num_exp=2, custom_metric_dict=None)
        b.benchmark_sample_complexity(T_list = [100, 500], num_vars=5, graph_density=0.1,\
                           fn = lambda x:x, coef=0.1, noise_fn=np.random.randn, data_max_lag=2)
        
        gt=[[{'PC-Pearson': {'time_taken': 0.26386189460754395,
            'precision': 0.6,
            'recall': 0.8,
            'f1_score': 0.6},
           'PC-Log-Likelihood': {'time_taken': 0.20469188690185547,
            'precision': 0.6,
            'recall': 0.8,
            'f1_score': 0.6},
           'PC-Mod-Log-Likelihood': {'time_taken': 6.858017921447754,
            'precision': 0.2,
            'recall': 0.8,
            'f1_score': 0.2},
           'PC-Freeman-Tukey': {'time_taken': 4.636682033538818,
            'precision': 0.0,
            'recall': 0.8,
            'f1_score': 0.0},
           'PC-Neyman': {'time_taken': 0.2073659896850586,
            'precision': 0.09999999500000026,
            'recall': 0.8999999950000003,
            'f1_score': 0.09999998500000128}},
          {'PC-Pearson': {'time_taken': 0.2293839454650879,
            'precision': 0.8,
            'recall': 1.0,
            'f1_score': 0.8},
           'PC-Log-Likelihood': {'time_taken': 0.33281517028808594,
            'precision': 0.6,
            'recall': 1.0,
            'f1_score': 0.6},
           'PC-Mod-Log-Likelihood': {'time_taken': 2.558051109313965,
            'precision': 0.0,
            'recall': 1.0,
            'f1_score': 0.0},
           'PC-Freeman-Tukey': {'time_taken': 2.6731669902801514,
            'precision': 0.0,
            'recall': 1.0,
            'f1_score': 0.0},
           'PC-Neyman': {'time_taken': 0.11328721046447754,
            'precision': 0.2,
            'recall': 1.0,
            'f1_score': 0.2}}],
         [{'PC-Pearson': {'time_taken': 0.1755049228668213,
            'precision': 0.8,
            'recall': 0.8,
            'f1_score': 0.8},
           'PC-Log-Likelihood': {'time_taken': 0.15719127655029297,
            'precision': 0.8,
            'recall': 0.8,
            'f1_score': 0.8},
           'PC-Mod-Log-Likelihood': {'time_taken': 0.8292937278747559,
            'precision': 0.6,
            'recall': 0.8,
            'f1_score': 0.6},
           'PC-Freeman-Tukey': {'time_taken': 0.7598118782043457,
            'precision': 0.6,
            'recall': 0.8,
            'f1_score': 0.6},
           'PC-Neyman': {'time_taken': 0.03947114944458008,
            'precision': 0.6,
            'recall': 0.8,
            'f1_score': 0.6}},
          {'PC-Pearson': {'time_taken': 0.1279439926147461,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'PC-Log-Likelihood': {'time_taken': 0.1361370086669922,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'PC-Mod-Log-Likelihood': {'time_taken': 0.14456391334533691,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'PC-Freeman-Tukey': {'time_taken': 0.11567997932434082,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'PC-Neyman': {'time_taken': 0.0393681526184082,
            'precision': 0.8,
            'recall': 1.0,
            'f1_score': 0.8}}]]
        
        for l1,l2 in zip(gt, b.results_full):
            for g1,g2 in zip(l1, l2):
                for algo in g1.keys():
                    for metric in ['precision', 'recall', 'f1_score']:
                        self.assertTrue(f'{g1[algo][metric]:.2f}'==f'{g2[algo][metric]:.2f}')

        
# test = TestBenchmarkDiscreteTimeSeries()
# test.test_benchmark_graph_density()
# test.test_benchmark_variable_complexity()
# test.test_benchmark_sample_complexity()
