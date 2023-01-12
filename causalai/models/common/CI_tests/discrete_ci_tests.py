from scipy import stats
import numpy as np
from numpy import ndarray
import pandas as pd
from typing import Tuple, List, Union, Optional

class DiscreteCI_tests:
    '''
    Performs CI test for discrete variables using the specified method. The null hypothesis
    for the test is X is independent of Y given Z.
    '''
    def __init__(self, method="pearson"):
        '''
        :param method: Options:

            - "pearson": "Chi-squared test"
            
            - "log-likelihood": "G-test or log-likelihood"
            
            - "freeman-tukey": "Freeman-Tukey Statistic"
            
            - "mod-log-likelihood": "Modified Log-likelihood"
            
            - "neyman": "Neyman's statistic"
        :type method: str
        '''
        self.method = method

    def run_test(self, x: ndarray, y: ndarray, z: Optional[ndarray]=None) -> Tuple[float,float]:
        """
        compute the test statistics and pvalues

        :param data_x: input data for x
        :type data_x: ndarray
        :param data_y: input data for y
        :type data_y: ndarray
        :param data_z: input data for z
        :type data_z: ndarray
        
        :return: Returns a tuple of 2 floats-- test statistic and the corresponding pvalue
        :rtype: tuple of floats
        """
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)

        if z is None:
            data = pd.DataFrame(np.hstack((x, y)), columns=['x', 'y'])
            val, p_value, dof, expected = stats.chi2_contingency(
                data.groupby(['x', 'y']).size().unstack('y', fill_value=0), lambda_=self.method
            )
        else:
            cond_names = [f'z{i}' for i in range(z.shape[1])]
            data = pd.DataFrame(np.hstack((x, y, z)), columns=['x', 'y', *cond_names])
            val = 0
            dof = 0
            cond_names = cond_names[0] if len(cond_names)==1 else cond_names
            for z_state, df in data.groupby(cond_names):
                try:
                    c, _, d, _ = stats.chi2_contingency(
                        df.groupby(['x', 'y']).size().unstack('y', fill_value=0), lambda_=self.method
                    )
                    val += c
                    dof += d
                except ValueError:
                    logging.info(f"Skipping a conditional test due to Nnt enough samples.")
            p_value = 1 - stats.chi2.cdf(val, df=dof)

        return val, p_value