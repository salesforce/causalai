
import unittest
import math
from causalai.benchmark.tabular.discrete import BenchmarkDiscreteTabular

from functools import partial
import numpy as np

def define_treatments(name, t,c):
    treatment = dict(var_name=name,
                    treatment_value=t,
                    control_value=c)
    return treatment

class TestBenchmarkDiscreteTabular(unittest.TestCase):
    def test_benchmark_graph_density(self):
        
        np.random.seed(0)
        b = BenchmarkDiscreteTabular(algo_dict=None, kargs_dict=None, 
                                     num_exp=2, custom_metric_dict=None)
        b.benchmark_graph_density(graph_density_list = [0.1, 0.5], num_vars=5, T=5000,\
                                   fn= lambda x:x, coef=0.1, noise_fn=np.random.rand)

        
        gt = [[{'PC-Pearson': {'time_taken': 0.053112030029296875,
            'precision': 0.7999999800000019,
            'recall': 0.8999999950000003,
            'f1_score': 0.7333333155555566},
           'PC-Log-Likelihood': {'time_taken': 0.043592214584350586,
            'precision': 0.7999999800000019,
            'recall': 0.8999999950000003,
            'f1_score': 0.7333333155555566},
           'PC-Mod-Log-Likelihood': {'time_taken': 0.03607010841369629,
            'precision': 0.7999999800000019,
            'recall': 0.8999999950000003,
            'f1_score': 0.7333333155555566},
           'PC-Freeman-Tukey': {'time_taken': 0.03241705894470215,
            'precision': 0.7999999800000019,
            'recall': 0.8999999950000003,
            'f1_score': 0.7333333155555566},
           'PC-Neyman': {'time_taken': 0.031136035919189453,
            'precision': 0.7999999800000019,
            'recall': 0.8999999950000003,
            'f1_score': 0.7333333155555566}},
          {'PC-Pearson': {'time_taken': 0.03594374656677246,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'PC-Log-Likelihood': {'time_taken': 0.031058073043823242,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'PC-Mod-Log-Likelihood': {'time_taken': 0.030583858489990234,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'PC-Freeman-Tukey': {'time_taken': 0.030870914459228516,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0},
           'PC-Neyman': {'time_taken': 0.030985116958618164,
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0}}],
         [{'PC-Pearson': {'time_taken': 0.25414395332336426,
            'precision': 0.5999999733333355,
            'recall': 0.5666666394444467,
            'f1_score': 0.5809523240362864},
           'PC-Log-Likelihood': {'time_taken': 0.23313283920288086,
            'precision': 0.5999999733333355,
            'recall': 0.5666666394444467,
            'f1_score': 0.5809523240362864},
           'PC-Mod-Log-Likelihood': {'time_taken': 0.21422600746154785,
            'precision': 0.5999999733333355,
            'recall': 0.5666666394444467,
            'f1_score': 0.5809523240362864},
           'PC-Freeman-Tukey': {'time_taken': 0.21939587593078613,
            'precision': 0.5999999733333355,
            'recall': 0.5666666394444467,
            'f1_score': 0.5809523240362864},
           'PC-Neyman': {'time_taken': 0.305621862411499,
            'precision': 0.4533333064888911,
            'recall': 0.6333333038888913,
            'f1_score': 0.4406014457691264}},
          {'PC-Pearson': {'time_taken': 0.06326413154602051,
            'precision': 0.49999995750000403,
            'recall': 0.999999940000006,
            'f1_score': 0.5333332577777834},
           'PC-Log-Likelihood': {'time_taken': 0.05014991760253906,
            'precision': 0.49999995750000403,
            'recall': 0.999999940000006,
            'f1_score': 0.5333332577777834},
           'PC-Mod-Log-Likelihood': {'time_taken': 0.04535508155822754,
            'precision': 0.49999995750000403,
            'recall': 0.999999940000006,
            'f1_score': 0.5333332577777834},
           'PC-Freeman-Tukey': {'time_taken': 0.04532289505004883,
            'precision': 0.49999995750000403,
            'recall': 0.999999940000006,
            'f1_score': 0.5333332577777834},
           'PC-Neyman': {'time_taken': 0.04535508155822754,
            'precision': 0.49999995750000403,
            'recall': 0.999999940000006,
            'f1_score': 0.5333332577777834}}]]
    
        for l1,l2 in zip(gt, b.results_full):
            for g1,g2 in zip(l1, l2):
                for algo in g1.keys():
                    for metric in ['precision', 'recall', 'f1_score']:
                        self.assertTrue(f'{g1[algo][metric]:.2f}'==f'{g2[algo][metric]:.2f}')
    
    def test_benchmark_variable_complexity(self):
        
        np.random.seed(0)
        b = BenchmarkDiscreteTabular(algo_dict=None, kargs_dict=None, 
                                     num_exp=2, custom_metric_dict=None)
        b.benchmark_variable_complexity(num_vars_list = [2,10], graph_density=0.1, T=1000,\
                           fn = lambda x:x, coef=0.1, noise_fn=np.random.randn)

        
        gt = [[{'PC-Pearson': {'time_taken': 0.00793313980102539,
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0},
               'PC-Log-Likelihood': {'time_taken': 0.006417036056518555,
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0},
               'PC-Mod-Log-Likelihood': {'time_taken': 0.00587010383605957,
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0},
               'PC-Freeman-Tukey': {'time_taken': 0.004995107650756836,
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0},
               'PC-Neyman': {'time_taken': 0.005011081695556641,
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0}},
              {'PC-Pearson': {'time_taken': 0.003912925720214844,
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0},
               'PC-Log-Likelihood': {'time_taken': 0.003986358642578125,
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0},
               'PC-Mod-Log-Likelihood': {'time_taken': 0.0042150020599365234,
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0},
               'PC-Freeman-Tukey': {'time_taken': 0.0050508975982666016,
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0},
               'PC-Neyman': {'time_taken': 0.0052032470703125,
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0}}],
             [{'PC-Pearson': {'time_taken': 0.15493035316467285,
                'precision': 0.49999999000000095,
                'recall': 0.6999999900000009,
                'f1_score': 0.49999998500000126},
               'PC-Log-Likelihood': {'time_taken': 0.12646818161010742,
                'precision': 0.49999999000000095,
                'recall': 0.6999999900000009,
                'f1_score': 0.49999998500000126},
               'PC-Mod-Log-Likelihood': {'time_taken': 0.14540624618530273,
                'precision': 0.46666666444444455,
                'recall': 0.6999999900000009,
                'f1_score': 0.4799999904000005},
               'PC-Freeman-Tukey': {'time_taken': 0.1272289752960205,
                'precision': 0.49999999000000095,
                'recall': 0.6999999900000009,
                'f1_score': 0.49999998500000126},
               'PC-Neyman': {'time_taken': 0.1559438705444336,
                'precision': 0.46666666444444455,
                'recall': 0.6999999900000009,
                'f1_score': 0.4799999904000005}},
              {'PC-Pearson': {'time_taken': 0.1252288818359375,
                'precision': 0.4,
                'recall': 0.8,
                'f1_score': 0.4},
               'PC-Log-Likelihood': {'time_taken': 0.12838506698608398,
                'precision': 0.4,
                'recall': 0.8,
                'f1_score': 0.4},
               'PC-Mod-Log-Likelihood': {'time_taken': 0.1253359317779541,
                'precision': 0.49999999000000095,
                'recall': 0.899999990000001,
                'f1_score': 0.49999998500000126},
               'PC-Freeman-Tukey': {'time_taken': 0.13924407958984375,
                'precision': 0.49999999000000095,
                'recall': 0.899999990000001,
                'f1_score': 0.49999998500000126},
               'PC-Neyman': {'time_taken': 0.15296006202697754,
                'precision': 0.5333333222222233,
                'recall': 0.9999999800000019,
                'f1_score': 0.5499999787500015}}]]
    
        for l1,l2 in zip(gt, b.results_full):
            for g1,g2 in zip(l1, l2):
                for algo in g1.keys():
                    for metric in ['precision', 'recall', 'f1_score']:
                        self.assertTrue(f'{g1[algo][metric]:.2f}'==f'{g2[algo][metric]:.2f}')
    
    def test_benchmark_sample_complexity(self):
        
        np.random.seed(0)
        b = BenchmarkDiscreteTabular(algo_dict=None, kargs_dict=None, 
                                     num_exp=2, custom_metric_dict=None)
        b.benchmark_sample_complexity(T_list = [100, 500,], num_vars=5, graph_density=0.1,\
                           fn = lambda x:x, coef=0.1, noise_fn=np.random.randn)
        
        gt = [[{'PC-Pearson': {'time_taken': 0.04793691635131836,
                'precision': 0.8,
                'recall': 0.8,
                'f1_score': 0.8},
               'PC-Log-Likelihood': {'time_taken': 0.04380202293395996,
                'precision': 0.8,
                'recall': 0.8,
                'f1_score': 0.8},
               'PC-Mod-Log-Likelihood': {'time_taken': 0.03721308708190918,
                'precision': 0.8,
                'recall': 0.8,
                'f1_score': 0.8},
               'PC-Freeman-Tukey': {'time_taken': 0.03280782699584961,
                'precision': 0.8,
                'recall': 0.8,
                'f1_score': 0.8},
               'PC-Neyman': {'time_taken': 0.027966976165771484,
                'precision': 0.399999980000002,
                'recall': 0.8999999950000003,
                'f1_score': 0.3333333155555566}},
              {'PC-Pearson': {'time_taken': 0.03661322593688965,
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0},
               'PC-Log-Likelihood': {'time_taken': 0.028981924057006836,
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0},
               'PC-Mod-Log-Likelihood': {'time_taken': 0.04886293411254883,
                'precision': 0.4,
                'recall': 1.0,
                'f1_score': 0.4},
               'PC-Freeman-Tukey': {'time_taken': 0.04340672492980957,
                'precision': 0.4,
                'recall': 1.0,
                'f1_score': 0.4},
               'PC-Neyman': {'time_taken': 0.36174607276916504,
                'precision': 0.0,
                'recall': 1.0,
                'f1_score': 0.0}}],
             [{'PC-Pearson': {'time_taken': 0.03754281997680664,
                'precision': 0.8,
                'recall': 0.8,
                'f1_score': 0.8},
               'PC-Log-Likelihood': {'time_taken': 0.033609867095947266,
                'precision': 0.8,
                'recall': 0.8,
                'f1_score': 0.8},
               'PC-Mod-Log-Likelihood': {'time_taken': 0.028799057006835938,
                'precision': 0.8,
                'recall': 0.8,
                'f1_score': 0.8},
               'PC-Freeman-Tukey': {'time_taken': 0.02888798713684082,
                'precision': 0.8,
                'recall': 0.8,
                'f1_score': 0.8},
               'PC-Neyman': {'time_taken': 0.030397891998291016,
                'precision': 0.8,
                'recall': 0.8,
                'f1_score': 0.8}},
              {'PC-Pearson': {'time_taken': 0.03240394592285156,
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0},
               'PC-Log-Likelihood': {'time_taken': 0.033824920654296875,
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0},
               'PC-Mod-Log-Likelihood': {'time_taken': 0.02854609489440918,
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0},
               'PC-Freeman-Tukey': {'time_taken': 0.0297391414642334,
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0},
               'PC-Neyman': {'time_taken': 0.031701087951660156,
                'precision': 1.0,
                'recall': 1.0,
                'f1_score': 1.0}}]]
        for l1,l2 in zip(gt, b.results_full):
            for g1,g2 in zip(l1, l2):
                for algo in g1.keys():
                    for metric in ['precision', 'recall', 'f1_score']:
                        self.assertTrue(f'{g1[algo][metric]:.2f}'==f'{g2[algo][metric]:.2f}')
        
        
# test = TestBenchmarkDiscreteTabular()
# test.test_benchmark_graph_density()
# test.test_benchmark_variable_complexity()
# test.test_benchmark_sample_complexity()
