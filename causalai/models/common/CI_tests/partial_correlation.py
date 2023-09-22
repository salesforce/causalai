
from scipy import stats
import numpy as np
from numpy import ndarray
from typing import Tuple, List, Union, Optional
import sys


class PartialCorrelation:
    '''
    Partial Correlation test for PC algorithm when causal links have linear dependency
    '''
    def __init__(self):
        pass

    def run_test(self, x: ndarray, y: ndarray, z: Optional[ndarray]=None) -> Tuple[float,float]:
        '''
        compute the test statistics and pvalues

        :param data_x: input data for x
        :type data_x: ndarray
        :param data_y: input data for y
        :type data_y: ndarray
        :param data_z: input data for z
        :type data_z: ndarray
        
        :return: Returns a tuple of 2 floats-- test statistic and the corresponding pvalue
        :rtype: tuple of floats
        '''

        self.x = x
        self.y = y
        self.z = z

        test_stat = self.get_correlation()
        pvalue = self.get_pvalue(test_stat)
        return test_stat, pvalue

    def get_correlation(self) -> float:
        '''
        pearson's correlation between residuals
        '''
        x_residual = self._get_residual_error(self.x)
        y_residual = self._get_residual_error(self.y)
        val, _ = stats.pearsonr(x_residual, y_residual)
        return val

    def _get_residual_error(self, v: ndarray) -> ndarray:
        """
        Returns residuals of linear regression. Performs a OLS regression over the given variable v given z.


        :param v: ndarray
            data array that is approximated by regression over self.z

        :return: ndarray
            resid : ndarray, the residual of the regression.
        """
        if self.z is not None:
            beta_hat = np.linalg.lstsq(self.z, v, rcond=None)[0]
            mean = np.dot(self.z, beta_hat)
            resid = v - mean
        else:
            resid = v
            mean = None
        return resid

    def get_pvalue(self, value: float) -> float:
        """
        See these links for the concept: 
        https://www.statology.org/p-value-correlation-excel/
        https://stats.libretexts.org/Bookshelves/Introductory_Statistics/Book%3A_Introductory_Statistics_(OpenStax)/12%3A_Linear_Regression_and_Correlation/12.05%3A_Testing_the_Significance_of_the_Correlation_Coefficient


        Why we use t-distribution and t-score for statistical significance and not Gaussian and z-score?
        https://www.jmp.com/en_us/statistics-knowledge-portal/t-test/t-distribution.html
        Basically, the standard normal or z-distribution assumes that you know the population standard deviation. The t-distribution is based on the sample standard deviation.
        When computing Pearson's correlation, we only have access to a scalar, which is one sample (from which distribution?).

        Returns analytic p-value from Student's t-test for the Pearson correlation coefficient.

        Assumes two-sided correlation. If the degrees of freedom are less than 1, numpy.nan is returned.

        The null hypothesis (large p-values) is that the correlation between x and y is not significantly different from 0.
        For a clear understanding, this means that when p-values are closer to 0, x and y are dependent, and independent otherwise.

        :param value: Test statistic value.
        :type value: float

        :return: Returns the pvalue. Larger p-values here indicate a larger likelihood of independence.
        :rtype: float
        """
        # Get the number of degrees of freedom
        dim = 2 + self.z.shape[1] if self.z is not None else 2 # z contains the condition set. Size = (num observations x num variables in the condition set)
        deg_freedom = self.x.shape[0] - dim

        if deg_freedom < 1:
            pval = np.nan
        elif abs(abs(value) - 1.0) <= sys.float_info.min:
            pval = 0.0
        else:
            t_score = value * np.sqrt(deg_freedom/(1. - value*value))
            # Two sided significance level
            pval = stats.t.sf(np.abs(t_score), deg_freedom) * 2

        return pval

