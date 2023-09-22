import unittest
from causalai.tests.data.transforms.networkx_helper_functions import *
from causalai.data.data_generator import DataGenerator, GenerateRandomTabularSEM
from causalai.models.tabular.grow_shrink import GrowShrink as GS
import random
from causalai.data.tabular import TabularData
from causalai.models.common.CI_tests.partial_correlation import PartialCorrelation
from causalai.models.common.CI_tests.kci import KCI
from causalai.models.common.CI_tests.discrete_ci_tests import DiscreteCI_tests
from causalai.models.common.CI_tests.ccit import CCITtest
import pandas as pd
import warnings


# testing parameters

num_vars = 20
max_num_parents = 4
num_samples = 5000
discrete = False
noise_fn = None #default noise, not noiseless
partial_knowledge_probabilities = [0.0, 0.5, 1.0]
use_multiprocessing = False
num_vars_to_check = 5
pvalue_thres = 0.05
CI_test_type = PartialCorrelation
update_shrink_options = {False, True}

vars = list(range(num_vars))
vars = [str(v) for v in vars]

vars_to_check = random.sample(vars, num_vars_to_check)

class TestGS(unittest.TestCase):

    def test_gc_perfect(self):

        warnings.simplefilter("ignore", ResourceWarning)

        #create model

        sem = GenerateRandomTabularSEM(var_names=vars, max_num_parents=max_num_parents)
        data, var_names, graph_gt = DataGenerator(sem, T=num_samples, discrete=discrete)
        data = TabularData(data = data, var_names = var_names)
        graph = causalai2networkx(graph_gt)
        full_knowledge = collect_all_knowledge(graph)
        priors = [None]+[get_partial_knowledge(graph, full_knowledge, inclusion_probability) for inclusion_probability in
                  partial_knowledge_probabilities]

        for prior in priors:
            for update_shrink in update_shrink_options:
                for var in vars_to_check:
                    perfect_ci_test = PerfectTest(graph, data)
                    gs_instance = GS(data = data, prior_knowledge = prior, CI_test=perfect_ci_test,
                    use_multiprocessing = use_multiprocessing, update_shrink = update_shrink)
                    mb = gs_instance.run(target_var=var, pvalue_thres = pvalue_thres)['markov_blanket']
                    mb = set(mb)
                    expected = compute_markov_blanket(graph, var)
                    self.assertEqual(expected, mb)

        warnings.simplefilter("default", ResourceWarning)

    def test_gc_imperfect(self):

        warnings.simplefilter("ignore", ResourceWarning)

        sem = GenerateRandomTabularSEM(var_names=vars, max_num_parents=max_num_parents)
        data, var_names, graph_gt = DataGenerator(sem, T=num_samples, discrete=discrete)
        data = TabularData(data=data, var_names = var_names)
        graph = causalai2networkx(graph_gt)
        full_knowledge = collect_all_knowledge(graph)
        priors = [None] + [get_partial_knowledge(graph, full_knowledge, inclusion_probability) for inclusion_probability
                           in
                           partial_knowledge_probabilities]
        print('prior', priors[2])

        for prior in priors:
            for update_shrink in update_shrink_options:
                results = pd.DataFrame(index=vars_to_check, columns=['tp', 'tn', 'fp', 'fn'])
                for var in vars_to_check:
                    gs_instance = GS(data = data, prior_knowledge = prior, CI_test=CI_test_type(),
                    use_multiprocessing = use_multiprocessing, update_shrink = update_shrink)
                    mb = gs_instance.run(target_var=var, pvalue_thres = pvalue_thres)['markov_blanket']
                    mb = set(mb)
                    expected = compute_markov_blanket(graph, var)
                    fp = mb - expected
                    fn = expected - mb
                    tp = expected.intersection(mb)
                    tn = ((set(vars) - expected)-mb)-{var}
                    results.loc[var, 'fp'] = len(fp)
                    results.loc[var, 'fn'] = len(fn)
                    results.loc[var, 'tp'] = len(tp)
                    results.loc[var, 'tn'] = len(tn)
                print(results.sum())

        warnings.simplefilter("default", ResourceWarning)





