'''

The Grow-Shrink algorithm can be used for discovering the minimal Markov blanket (MB) of a target variable in tabular
data. A MB is a minimal conditioning set making the target variable independent of all other variables; under the
assumption of faithfulness, which we make here, the MB is unique and corresponds to the set of parents, children and
co-parents of the target variable. The MB can be used for feature selection.

The Grow-Shrink algorithm operates in two phases, called growth and shrink. The growth phase first adds to the MB
estimation variables unconditionally dependent on the target variable, then conditions on those variables and adds the
conditionally dependent variables to the estimation. Assuming perfect conditional independence testing, this yields a
superset of the actual MB. The shrink phase then removes from the estimated MB variables independent from the target
variable conditional on all other variables in the MB estimation. The algorithm does not partition the estimated MB into
parents/children/co-parents.

The assumptions we make for the growth-shrink algorithm are: 1. Causal Markov condition, which implies that two
variables that are d-separated in a causal graph are probabilistically independent, 2. faithfulness, i.e., no
conditional independence can hold unless the Causal Markov condition is met, 3. no hidden confounders, and 4. no cycles
in the causal graph.

'''

from typing import TypedDict, Tuple, List, Union, Optional, Dict, Callable
import copy
from collections import defaultdict
try:
    import ray
except:
    pass
from .base import BaseTabularAlgo, ResultInfoTabularMB
from ...data.tabular import TabularData
from ...models.common.prior_knowledge import PriorKnowledge
from ...models.common.CI_tests.partial_correlation import PartialCorrelation
from ...models.common.CI_tests.kci import KCI
from ...models.common.CI_tests.discrete_ci_tests import DiscreteCI_tests
from ...models.common.CI_tests.ccit import CCITtest


class GrowShrink(BaseTabularAlgo):
    '''
    Grow-Shrink (GS) algorithm for estimating a minimal markov blanket in tabular data. For details, see:
    "Bayesian Network Induction via Local Neighborhoods", Dimitris Margaritis and Sebastian Thrun, NeurIPS 1999.
    '''

    def __init__(self, data: TabularData, prior_knowledge: Optional[PriorKnowledge] = None,
                 CI_test: Union[PartialCorrelation, KCI, DiscreteCI_tests, CCITtest] = PartialCorrelation(),
                 use_multiprocessing: Optional[bool] = False, update_shrink: bool = False):
        '''
        Grow-Shrink (GS) algorithm for estimating a minimal markov blanket.

        :param data: It contains data.values, a numpy array of shape (observations N, variables D).
        :type data: TabularData object
        :param prior_knowledge: Specify prior knowledge to the causal discovery process by either
            forbidding links/co-parents that are known to not exist, or adding back links/co-parents that do exist
            based on expert knowledge. See the PriorKnowledge class for more details.
        :type prior_knowledge: PriorKnowledge object
        :param CI_test: This object perform conditional independence tests (default: PartialCorrelation).
            See object class for more details.
        :type CI_test: PartialCorrelation, KCI, or CCITtest object
        :param use_multiprocessing: If True, computations are performed using multi-processing which makes the algorithm
            faster.
        :type use_multiprocessing: bool
        :param update_shrink: whether to update the markov blanket during the shrink phase or not. update_shrink=True
            reduces the size of the conditioning sets tested (which usually increases the quality of the CI test), but makes
            the algorithm susceptible to cumulative error when a variable from the minimal markov blanket is mistakenly
            removed due to previous error of the CI test. Note: this option disables multiprocessing at the shrink phase.
        :type update_shrink: bool
        '''
        BaseTabularAlgo.__init__(self, data=data, prior_knowledge=prior_knowledge, CI_test=CI_test,
                                 use_multiprocessing=use_multiprocessing, update_shrink=update_shrink)
        self.CI_test_ = lambda x, y, z: CI_test.run_test(x, y, z)
        if use_multiprocessing:
            if 'ray' in globals():
                self.CI_test_remote_ = ray.remote(self.CI_test_)  # Ray wrapper; avoiding Ray Actors because they are slower
            else:
                print(
                    'use_multiprocessing was specified as True but cannot be used because the ray library is not installed. Install using pip install ray.')

    def run(self, target_var: Union[int, str], pvalue_thres: float = 0.05) -> ResultInfoTabularMB:
        """
        Runs GS algorithm for estimating markov blnaket.

        :param target_var: Target variable index for which parents need to be estimated.
        :type target_var: int
        :param pvalue_thres: Significance level used for hypothesis testing (default: 0.05). Candidate variable with pvalues above pvalue_thres
            are ignored, and the rest are returned as the markov blanket of the target_var.
        :type pvalue_thres: float

        :return: Dictionary has three keys:

            - markov_blanket : List of estimated markov blanket variables.

            - value_dict : Dictionary of form {var3_name:float, ...} containing the test statistic of a link.

            - pvalue_dict : Dictionary of form {var3_name:float, ...} containing the p-value corresponding to the above test statistic.
        :rtype: dict
        """

        assert target_var in self.data.var_names, f'{target_var} not found in the variable names specified for the data!'

        if (self.use_multiprocessing == True) and ('ray' in globals()):
            self.start()

        data = self.data

        record_values = defaultdict(list)
        record_p_values = defaultdict(list)

        required_mb = self.prior_knowledge.required(target_var, 'included')
        required_not_mb = self.prior_knowledge.required(target_var, 'excluded')
        # Remaining variables all begin outside of the estimated markov blanket
        tbd_not_mb = list(set(self.get_candidate_mb(target_var)).difference(required_mb, required_not_mb))

        tbd_mb = []

        mb = required_mb + tbd_mb
        not_mb = required_not_mb + tbd_not_mb

        mb_change = True  # tracks whether a new variable has been added to mb
        #target_var_data = data[:, target_var]

        # growth phase
        counter = 0  # with a perfect CI-test, after 2 growth phase iterations, minimal markov blanket is included in mb
        while (mb_change and counter < 2):
            counter += 1
            mb_change = False
            #mb_data = None if mb == [] else data[:, mb]

            if (self.use_multiprocessing == True) and ('ray' in globals()):
                ci_dict = {candidate: self.CI_test_remote_.remote(*data.extract_array(candidate, target_var, mb)) for candidate in tbd_not_mb}
                ci_dict = {candidate: ray.get(ci_dict[candidate]) for candidate in tbd_not_mb}
            else:
                ci_dict = {candidate: self.CI_test_(*data.extract_array(candidate, target_var, mb)) for
                           candidate in tbd_not_mb}
            for candidate in copy.copy(tbd_not_mb):  # tbd_not_mb needs to be copied as it is mutated in the loop
                value, pvalue = ci_dict[candidate]
                record_values[candidate].append(value)
                record_p_values[candidate].append(pvalue)
                if pvalue < pvalue_thres:
                    tbd_mb.append(candidate)
                    tbd_not_mb.remove(candidate)
                    mb_change = True
            mb = required_mb + tbd_mb
            not_mb = required_not_mb + tbd_not_mb

        # shrink phase
        old_tbd_mb = copy.copy(tbd_mb)

        if not self.update_shrink:

            ci_dict = {}
            for candidate in tbd_mb:
                mb_except_candidate = copy.copy(mb)
                mb_except_candidate.remove(candidate)
                if (self.use_multiprocessing == True) and ('ray' in globals()):
                    ci_dict[candidate] = self.CI_test_remote_.remote(*data.extract_array(candidate, target_var, mb_except_candidate))
                else:
                    ci_dict[candidate] = self.CI_test_(*data.extract_array(candidate, target_var, mb_except_candidate))
            if (self.use_multiprocessing == True) and ('ray' in globals()):
                ci_dict = {candidate: ray.get(ci_dict[candidate]) for candidate in tbd_mb}

            for candidate in tbd_mb:
                value, pvalue = ci_dict[candidate][0], ci_dict[candidate][1]
                record_values[candidate].append(value)
                record_p_values[candidate].append(pvalue)

            for candidate in old_tbd_mb:
                if not ci_dict[candidate][1] < pvalue_thres:
                    tbd_not_mb.append(candidate)
                    tbd_mb.remove(candidate)

            mb = required_mb + tbd_mb
            not_mb = required_not_mb + tbd_not_mb

        else:

            for candidate in old_tbd_mb:

                considered_mb = copy.copy(mb)
                mb_except_candidate = copy.copy(considered_mb)
                mb_except_candidate.remove(candidate)
                value, pvalue = self.CI_test_(*data.extract_array(candidate, target_var, mb))
                record_values[candidate].append(value)
                record_p_values[candidate].append(pvalue)
                if not pvalue < pvalue_thres:
                    tbd_not_mb.append(candidate)
                    tbd_mb.remove(candidate)
                    mb = required_mb + tbd_mb
                    not_mb = required_not_mb + tbd_not_mb

        if (self.use_multiprocessing == True) and ('ray' in globals()):
            self.stop()

        self.value_dict = {var: values[-1] for var, values in record_values.items()}
        self.pvalue_dict = {var: pvalues[-1] for var, pvalues in record_p_values.items()}
        self.full_record_values = record_values
        self.full_record_pvalues = record_p_values

        self.result = {'markov_blanket': mb,
                       'value_dict': self.value_dict,
                       'pvalue_dict': self.pvalue_dict,
                       'full_record_values': self.full_record_values,
                       'full_record_pvalues': self.full_record_pvalues}

        return self.result