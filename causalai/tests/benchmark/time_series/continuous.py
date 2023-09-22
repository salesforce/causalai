
import unittest
import math
from causalai.benchmark.time_series.continuous import BenchmarkContinuousTimeSeries

from functools import partial
import numpy as np

def define_treatments(name, t,c):
    treatment = dict(var_name=name,
                    treatment_value=t,
                    control_value=c)
    return treatment

class TestBenchmarkContinuousTimeSeries(unittest.TestCase):
    def test_benchmark_graph_density(self):
        
        np.random.seed(0)
        b = BenchmarkContinuousTimeSeries(algo_dict=None, kargs_dict=None, 
                                     num_exp=2, custom_metric_dict=None)
        b.benchmark_graph_density(graph_density_list = [0.1, 0.5], num_vars=5, T=5000,\
                                   fn= lambda x:x, coef=0.1, noise_fn=np.random.rand, data_max_lag=2)

        gt=[[{'PC-PartialCorr': {'time_taken': 0.11409497261047363,
            'precision': 0.7999999900000005,
            'recall': 0.9999999900000006,
            'f1_score': 0.799999980000001},
           'Granger': {'time_taken': 0.326251745223999,
            'precision': 0.9999999900000006,
            'recall': 0.9999999900000006,
            'f1_score': 0.999999980000001},
           'VARLINGAM': {'time_taken': 0.10383296012878418,
            'precision': 0.9333333288888891,
            'recall': 0.9999999900000006,
            'f1_score': 0.9599999840000008}},
          {'PC-PartialCorr': {'time_taken': 0.0737142562866211,
            'precision': 0.8,
            'recall': 1.0,
            'f1_score': 0.8},
           'Granger': {'time_taken': 0.3537769317626953,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'VARLINGAM': {'time_taken': 0.11174702644348145,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0}}],
         [{'PC-PartialCorr': {'time_taken': 0.12348294258117676,
            'precision': 0.9333332822222266,
            'recall': 0.9999999433333381,
            'f1_score': 0.9599999073333398},
           'Granger': {'time_taken': 0.367264986038208,
            'precision': 0.9999999433333381,
            'recall': 0.9999999433333381,
            'f1_score': 0.9999999033333401},
           'VARLINGAM': {'time_taken': 0.11319708824157715,
            'precision': 0.9999999433333381,
            'recall': 0.9999999433333381,
            'f1_score': 0.9999999033333401}},
          {'PC-PartialCorr': {'time_taken': 0.08943676948547363,
            'precision': 0.8999999550000043,
            'recall': 0.999999940000006,
            'f1_score': 0.9333332555555616},
           'Granger': {'time_taken': 0.362746000289917,
            'precision': 0.999999940000006,
            'recall': 0.999999940000006,
            'f1_score': 0.9999999100000074},
           'VARLINGAM': {'time_taken': 0.10984396934509277,
            'precision': 0.999999940000006,
            'recall': 0.999999940000006,
            'f1_score': 0.9999999100000074}}]]
    
        for l1,l2 in zip(gt, b.results_full):
            for g1,g2 in zip(l1, l2):
                for algo in g1.keys():
                    for metric in ['precision', 'recall', 'f1_score']:
                        self.assertTrue(f'{g1[algo][metric]:.2f}'==f'{g2[algo][metric]:.2f}')
    
    def test_benchmark_variable_complexity(self):
        
        np.random.seed(0)
        b = BenchmarkContinuousTimeSeries(algo_dict=None, kargs_dict=None, 
                                     num_exp=2, custom_metric_dict=None)
        b.benchmark_variable_complexity(num_vars_list = [2,10], graph_density=0.1, T=1000,\
                           fn = lambda x:x, coef=0.1, noise_fn=np.random.randn, data_max_lag=2)

        gt=[[{'PC-PartialCorr': {'time_taken': 0.021373987197875977,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'Granger': {'time_taken': 0.09624814987182617,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'VARLINGAM': {'time_taken': 0.010593891143798828,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0}},
          {'PC-PartialCorr': {'time_taken': 0.013184070587158203,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'Granger': {'time_taken': 0.07614779472351074,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'VARLINGAM': {'time_taken': 0.0106201171875,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0}}],
         [{'PC-PartialCorr': {'time_taken': 0.33800292015075684,
            'precision': 0.9499999725000023,
            'recall': 0.9999999650000031,
            'f1_score': 0.9666666177777813},
           'Granger': {'time_taken': 0.5702190399169922,
            'precision': 0.6999999900000009,
            'recall': 0.6499999975000001,
            'f1_score': 0.6666666577777783},
           'VARLINGAM': {'time_taken': 0.25424790382385254,
            'precision': 0.9999999650000031,
            'recall': 0.9999999650000031,
            'f1_score': 0.9999999450000043}},
          {'PC-PartialCorr': {'time_taken': 0.2967500686645508,
            'precision': 0.7,
            'recall': 0.8,
            'f1_score': 0.7},
           'Granger': {'time_taken': 0.5952730178833008,
            'precision': 0.8,
            'recall': 0.8,
            'f1_score': 0.8},
           'VARLINGAM': {'time_taken': 0.2714519500732422,
            'precision': 0.8,
            'recall': 0.8,
            'f1_score': 0.8}}]]
        
        for l1,l2 in zip(gt, b.results_full):
            for g1,g2 in zip(l1, l2):
                for algo in g1.keys():
                    for metric in ['precision', 'recall', 'f1_score']:
                        self.assertTrue(f'{g1[algo][metric]:.2f}'==f'{g2[algo][metric]:.2f}')
    
    def test_benchmark_sample_complexity(self):
        
        np.random.seed(0)
        b = BenchmarkContinuousTimeSeries(algo_dict=None, kargs_dict=None, 
                                     num_exp=2, custom_metric_dict=None)
        b.benchmark_sample_complexity(T_list = [100, 500,], num_vars=5, graph_density=0.1,\
                           fn = lambda x:x, coef=0.1, noise_fn=np.random.randn, data_max_lag=2)
        
        gt=[[{'PC-PartialCorr': {'time_taken': 0.1007390022277832,
            'precision': 0.6,
            'recall': 0.8,
            'f1_score': 0.6},
           'Granger': {'time_taken': 0.18874788284301758,
            'precision': 0.8,
            'recall': 0.8,
            'f1_score': 0.8},
           'VARLINGAM': {'time_taken': 0.03464174270629883,
            'precision': 0.8,
            'recall': 0.8,
            'f1_score': 0.8}},
          {'PC-PartialCorr': {'time_taken': 0.06741619110107422,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'Granger': {'time_taken': 0.1784060001373291,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'VARLINGAM': {'time_taken': 0.031159162521362305,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0}}],
         [{'PC-PartialCorr': {'time_taken': 0.07787013053894043,
            'precision': 0.8,
            'recall': 0.8,
            'f1_score': 0.8},
           'Granger': {'time_taken': 0.18530607223510742,
            'precision': 0.8,
            'recall': 0.8,
            'f1_score': 0.8},
           'VARLINGAM': {'time_taken': 0.03673601150512695,
            'precision': 0.8,
            'recall': 0.8,
            'f1_score': 0.8}},
          {'PC-PartialCorr': {'time_taken': 0.0674741268157959,
            'precision': 0.8,
            'recall': 1.0,
            'f1_score': 0.8},
           'Granger': {'time_taken': 0.17926979064941406,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'VARLINGAM': {'time_taken': 0.03651309013366699,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0}}]]
        
        for l1,l2 in zip(gt, b.results_full):
            for g1,g2 in zip(l1, l2):
                for algo in g1.keys():
                    for metric in ['precision', 'recall', 'f1_score']:
                        self.assertTrue(f'{g1[algo][metric]:.2f}'==f'{g2[algo][metric]:.2f}')
        
    def test_benchmark_noise_type(self):
        
        np.random.seed(0)
        b = BenchmarkContinuousTimeSeries(algo_dict=None, kargs_dict=None, 
                                     num_exp=2, custom_metric_dict=None)
        b.benchmark_noise_type(noise_fn_types=[np.random.randn, np.random.rand], 
                            noise_fn_names=['Gaussian', 'Uniform'],
                            num_vars=5, graph_density=0.1, T=1000, fn = lambda x:x, 
                               coef=0.1, data_max_lag=2)

        gt=[[{'PC-PartialCorr': {'time_taken': 0.11704111099243164,
            'precision': 0.9333333288888891,
            'recall': 0.9999999900000006,
            'f1_score': 0.9599999840000008},
           'Granger': {'time_taken': 0.2468862533569336,
            'precision': 0.9999999800000019,
            'recall': 0.8999999950000003,
            'f1_score': 0.9333333155555567},
           'VARLINGAM': {'time_taken': 0.058667898178100586,
            'precision': 0.9999999900000006,
            'recall': 0.9999999900000006,
            'f1_score': 0.999999980000001}},
          {'PC-PartialCorr': {'time_taken': 0.0809028148651123,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'Granger': {'time_taken': 0.25846314430236816,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'VARLINGAM': {'time_taken': 0.06212186813354492,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0}}],
         [{'PC-PartialCorr': {'time_taken': 0.08890199661254883,
            'precision': 0.599999980000002,
            'recall': 0.8999999950000003,
            'f1_score': 0.5333333155555566},
           'Granger': {'time_taken': 0.2696869373321533,
            'precision': 0.8,
            'recall': 0.8,
            'f1_score': 0.8},
           'VARLINGAM': {'time_taken': 0.06480717658996582,
            'precision': 0.9999999800000019,
            'recall': 0.8999999950000003,
            'f1_score': 0.9333333155555567}},
          {'PC-PartialCorr': {'time_taken': 0.08728504180908203,
            'precision': 0.8,
            'recall': 1.0,
            'f1_score': 0.8},
           'Granger': {'time_taken': 0.26751160621643066,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'VARLINGAM': {'time_taken': 0.06279897689819336,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0}}]]
        
    
        for l1,l2 in zip(gt, b.results_full):
            for g1,g2 in zip(l1, l2):
                for algo in g1.keys():
                    for metric in ['precision', 'recall', 'f1_score']:
                        self.assertTrue(f'{g1[algo][metric]:.2f}'==f'{g2[algo][metric]:.2f}')
    
    def test_benchmark_snr(self):
        
        np.random.seed(0)
        b = BenchmarkContinuousTimeSeries(algo_dict=None, kargs_dict=None, 
                                     num_exp=2, custom_metric_dict=None)
        
        
        b.benchmark_snr(snr_list=[0.01, 0.1,], num_vars=5, graph_density=0.1, T=1000,\
                           fn = lambda x:x, noise_fn=np.random.randn, data_max_lag=2)
        
        gt=[[{'PC-PartialCorr': {'time_taken': 0.10017704963684082,
            'precision': 0.8,
            'recall': 0.8,
            'f1_score': 0.8},
           'Granger': {'time_taken': 0.24199199676513672,
            'precision': 0.8,
            'recall': 0.8,
            'f1_score': 0.8},
           'VARLINGAM': {'time_taken': 0.05918478965759277,
            'precision': 0.8,
            'recall': 0.8,
            'f1_score': 0.8}},
          {'PC-PartialCorr': {'time_taken': 0.0795738697052002,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'Granger': {'time_taken': 0.25279831886291504,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'VARLINGAM': {'time_taken': 0.06063103675842285,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0}}],
         [{'PC-PartialCorr': {'time_taken': 0.09854602813720703,
            'precision': 0.9333333288888891,
            'recall': 0.9999999900000006,
            'f1_score': 0.9599999840000008},
           'Granger': {'time_taken': 0.26831817626953125,
            'precision': 0.9999999800000019,
            'recall': 0.8999999950000003,
            'f1_score': 0.9333333155555567},
           'VARLINGAM': {'time_taken': 0.06113314628601074,
            'precision': 0.9999999900000006,
            'recall': 0.9999999900000006,
            'f1_score': 0.999999980000001}},
          {'PC-PartialCorr': {'time_taken': 0.07863807678222656,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'Granger': {'time_taken': 0.26860499382019043,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'VARLINGAM': {'time_taken': 0.0655980110168457,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0}}]]
        
        for l1,l2 in zip(gt, b.results_full):
            for g1,g2 in zip(l1, l2):
                for algo in g1.keys():
                    for metric in ['precision', 'recall', 'f1_score']:
                        self.assertTrue(f'{g1[algo][metric]:.2f}'==f'{g2[algo][metric]:.2f}')
        
# test = TestBenchmarkContinuousTimeSeries()
# test.test_benchmark_graph_density()
# test.test_benchmark_variable_complexity()
# test.test_benchmark_sample_complexity()
# test.test_benchmark_noise_type()
# test.test_benchmark_snr()
