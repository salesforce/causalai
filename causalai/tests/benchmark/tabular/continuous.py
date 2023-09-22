
import unittest
import math
from causalai.benchmark.tabular.continuous import BenchmarkContinuousTabular
from functools import partial
import numpy as np

def define_treatments(name, t,c):
    treatment = dict(var_name=name,
                    treatment_value=t,
                    control_value=c)
    return treatment

class TestBenchmarkContinuousTabular(unittest.TestCase):
    def test_benchmark_graph_density(self):
        
        np.random.seed(0)
        b = BenchmarkContinuousTabular(algo_dict=None, kargs_dict=None, 
                                     num_exp=2, custom_metric_dict=None)
        b.benchmark_graph_density(graph_density_list = [0.1, 0.5], num_vars=5, T=5000,\
                                   fn= lambda x:x, coef=0.1, noise_fn=np.random.rand)

        
        gt=[[{'PC-PartialCorr': {'time_taken': 0.02721095085144043,
            'precision': 0.5999999950000001,
            'recall': 0.9999999900000006,
            'f1_score': 0.5999999825000009},
           'GES': {'time_taken': 0.01672816276550293,
            'precision': 0.9999999900000006,
            'recall': 0.9999999900000006,
            'f1_score': 0.999999980000001},
           'LINGAM': {'time_taken': 0.06855487823486328,
            'precision': 0.9999999900000006,
            'recall': 0.9999999900000006,
            'f1_score': 0.999999980000001}},
          {'PC-PartialCorr': {'time_taken': 0.02438497543334961,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'GES': {'time_taken': 0.006247997283935547,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'LINGAM': {'time_taken': 0.07087206840515137,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0}}],
         [{'PC-PartialCorr': {'time_taken': 0.0814981460571289,
            'precision': 0.5499999714583356,
            'recall': 0.9999999433333381,
            'f1_score': 0.6314284994591881},
           'GES': {'time_taken': 0.04037785530090332,
            'precision': 0.46666663777778006,
            'recall': 0.7333332988888915,
            'f1_score': 0.49333327288889306},
           'LINGAM': {'time_taken': 0.06008505821228027,
            'precision': 0.9999999433333381,
            'recall': 0.9999999433333381,
            'f1_score': 0.9999999033333401}},
          {'PC-PartialCorr': {'time_taken': 0.020832061767578125,
            'precision': 0.49999995750000403,
            'recall': 0.999999940000006,
            'f1_score': 0.5333332577777834},
           'GES': {'time_taken': 0.01822209358215332,
            'precision': 0.49999995500000427,
            'recall': 0.999999940000006,
            'f1_score': 0.5333332555555615},
           'LINGAM': {'time_taken': 0.05796194076538086,
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
        b = BenchmarkContinuousTabular(algo_dict=None, kargs_dict=None, 
                                     num_exp=2, custom_metric_dict=None)
        b.benchmark_variable_complexity(num_vars_list = [2,10], graph_density=0.1, T=1000,\
                           fn = lambda x:x, coef=0.1, noise_fn=np.random.randn)

        gt=[[{'PC-PartialCorr': {'time_taken': 0.0031690597534179688,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'GES': {'time_taken': 0.0011239051818847656,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'LINGAM': {'time_taken': 0.0065228939056396484,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0}},
          {'PC-PartialCorr': {'time_taken': 0.0027151107788085938,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'GES': {'time_taken': 0.001155853271484375,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'LINGAM': {'time_taken': 0.006453990936279297,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0}}],
         [{'PC-PartialCorr': {'time_taken': 0.1103048324584961,
            'precision': 0.599999970000003,
            'recall': 0.849999977500002,
            'f1_score': 0.5666666277777808},
           'GES': {'time_taken': 0.14372491836547852,
            'precision': 0.6999999750000022,
            'recall': 0.8999999750000022,
            'f1_score': 0.699999960000003},
           'LINGAM': {'time_taken': 0.20328712463378906,
            'precision': 0.4,
            'recall': 0.6,
            'f1_score': 0.4}},
          {'PC-PartialCorr': {'time_taken': 0.08287787437438965,
            'precision': 0.849999988750001,
            'recall': 0.9999999800000019,
            'f1_score': 0.8666666438888905},
           'GES': {'time_taken': 0.06374073028564453,
            'precision': 0.849999987500001,
            'recall': 0.9999999800000019,
            'f1_score': 0.8666666427777795},
           'LINGAM': {'time_taken': 0.2053987979888916,
            'precision': 0.9999999800000019,
            'recall': 0.9999999800000019,
            'f1_score': 0.9999999700000025}}]]
    
        for l1,l2 in zip(gt, b.results_full):
            for g1,g2 in zip(l1, l2):
                for algo in g1.keys():
                    for metric in ['precision', 'recall', 'f1_score']:
                        self.assertTrue(f'{g1[algo][metric]:.2f}'==f'{g2[algo][metric]:.2f}')
    
    def test_benchmark_sample_complexity(self):
        
        np.random.seed(0)
        b = BenchmarkContinuousTabular(algo_dict=None, kargs_dict=None, 
                                     num_exp=2, custom_metric_dict=None)
        b.benchmark_sample_complexity(T_list = [100, 500,], num_vars=5, graph_density=0.1,\
                           fn = lambda x:x, coef=0.1, noise_fn=np.random.randn)
        
        gt=[[{'PC-PartialCorr': {'time_taken': 0.03637242317199707,
            'precision': 0.8,
            'recall': 0.8,
            'f1_score': 0.8},
           'GES': {'time_taken': 0.012650012969970703,
            'precision': 0.4,
            'recall': 0.8,
            'f1_score': 0.4},
           'LINGAM': {'time_taken': 0.054818153381347656,
            'precision': 0.8,
            'recall': 0.8,
            'f1_score': 0.8}},
          {'PC-PartialCorr': {'time_taken': 0.025879859924316406,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'GES': {'time_taken': 0.010455131530761719,
            'precision': 0.6,
            'recall': 1.0,
            'f1_score': 0.6},
           'LINGAM': {'time_taken': 0.026741981506347656,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0}}],
         [{'PC-PartialCorr': {'time_taken': 0.018736839294433594,
            'precision': 0.7999999800000019,
            'recall': 0.8999999950000003,
            'f1_score': 0.7333333155555566},
           'GES': {'time_taken': 0.013829231262207031,
            'precision': 0.9999999900000006,
            'recall': 0.9999999900000006,
            'f1_score': 0.999999980000001},
           'LINGAM': {'time_taken': 0.029428958892822266,
            'precision': 0.9999999800000019,
            'recall': 0.8999999950000003,
            'f1_score': 0.9333333155555567}},
          {'PC-PartialCorr': {'time_taken': 0.01752185821533203,
            'precision': 0.6,
            'recall': 1.0,
            'f1_score': 0.6},
           'GES': {'time_taken': 0.008615970611572266,
            'precision': 0.6,
            'recall': 1.0,
            'f1_score': 0.6},
           'LINGAM': {'time_taken': 0.028570890426635742,
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
        b = BenchmarkContinuousTabular(algo_dict=None, kargs_dict=None, 
                                     num_exp=2, custom_metric_dict=None)
        b.benchmark_noise_type(noise_fn_types=[np.random.randn, np.random.rand], 
                            noise_fn_names=['Gaussian', 'Uniform'],
                            num_vars=5, graph_density=0.1, T=1000, fn = lambda x:x, 
                               coef=0.1)

        gt=[[{'PC-PartialCorr': {'time_taken': 0.03591036796569824,
            'precision': 0.5999999950000001,
            'recall': 0.9999999900000006,
            'f1_score': 0.5999999825000009},
           'GES': {'time_taken': 0.01631474494934082,
            'precision': 0.9999999900000006,
            'recall': 0.9999999900000006,
            'f1_score': 0.999999980000001},
           'LINGAM': {'time_taken': 0.04410409927368164,
            'precision': 0.7999999800000019,
            'recall': 0.8999999950000003,
            'f1_score': 0.7333333155555566}},
          {'PC-PartialCorr': {'time_taken': 0.02033519744873047,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'GES': {'time_taken': 0.004396200180053711,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'LINGAM': {'time_taken': 0.033356666564941406,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0}}],
         [{'PC-PartialCorr': {'time_taken': 0.02126002311706543,
            'precision': 0.5999999950000001,
            'recall': 0.9999999900000006,
            'f1_score': 0.5999999825000009},
           'GES': {'time_taken': 0.013009786605834961,
            'precision': 0.5999999900000005,
            'recall': 0.9999999900000006,
            'f1_score': 0.599999980000001},
           'LINGAM': {'time_taken': 0.032029151916503906,
            'precision': 0.9999999900000006,
            'recall': 0.9999999900000006,
            'f1_score': 0.999999980000001}},
          {'PC-PartialCorr': {'time_taken': 0.016668081283569336,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'GES': {'time_taken': 0.00424504280090332,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'LINGAM': {'time_taken': 0.031610727310180664,
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
        b = BenchmarkContinuousTabular(algo_dict=None, kargs_dict=None, 
                                     num_exp=2, custom_metric_dict=None)
        
        
        b.benchmark_snr(snr_list=[0.01, 0.1,], num_vars=5, graph_density=0.1, T=1000,\
                           fn = lambda x:x, noise_fn=np.random.randn)
        
        gt=[[{'PC-PartialCorr': {'time_taken': 0.031065940856933594,
            'precision': 0.8,
            'recall': 0.8,
            'f1_score': 0.8},
           'GES': {'time_taken': 0.005977153778076172,
            'precision': 0.8,
            'recall': 0.8,
            'f1_score': 0.8},
           'LINGAM': {'time_taken': 0.04601693153381348,
            'precision': 0.8,
            'recall': 0.8,
            'f1_score': 0.8}},
          {'PC-PartialCorr': {'time_taken': 0.021618127822875977,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'GES': {'time_taken': 0.004547834396362305,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'LINGAM': {'time_taken': 0.032689809799194336,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0}}],
         [{'PC-PartialCorr': {'time_taken': 0.019637107849121094,
            'precision': 0.5999999950000001,
            'recall': 0.9999999900000006,
            'f1_score': 0.5999999825000009},
           'GES': {'time_taken': 0.012834787368774414,
            'precision': 0.9999999900000006,
            'recall': 0.9999999900000006,
            'f1_score': 0.999999980000001},
           'LINGAM': {'time_taken': 0.032476186752319336,
            'precision': 0.7999999800000019,
            'recall': 0.8999999950000003,
            'f1_score': 0.7333333155555566}},
          {'PC-PartialCorr': {'time_taken': 0.016483068466186523,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'GES': {'time_taken': 0.004041910171508789,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'LINGAM': {'time_taken': 0.03161025047302246,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0}}]]
        
        for l1,l2 in zip(gt, b.results_full):
            for g1,g2 in zip(l1, l2):
                for algo in g1.keys():
                    for metric in ['precision', 'recall', 'f1_score']:
                        self.assertTrue(f'{g1[algo][metric]:.2f}'==f'{g2[algo][metric]:.2f}')
        
# test = TestBenchmarkContinuousTabular()
# test.test_benchmark_graph_density()
# test.test_benchmark_variable_complexity()
# test.test_benchmark_sample_complexity()
# test.test_benchmark_noise_type()
# test.test_benchmark_snr()
