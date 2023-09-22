from CCIT import CCIT
from typing import List
import numpy as np

class CCITtest():
    '''
    A wrapper for Classifier Conditional Independence Test (CCIT). For details, see https://github.com/rajatsen91/CCIT.
    '''
    def __init__(self, max_depths: List[int] = [6, 10, 13], n_estimators: List[int] = [100, 200, 300],
                 colsample_bytrees: List[float] = [0.8], nfold: int = 5, feature_selection: int = 0,
                 train_samp: int = -1, k: int = 1, threshold: float = 0.03, num_iter: int = 30, bootstrap: bool = True,
                 nthread: int = 8):
        '''
        Constructor. Inputs are saved as attributes under keyword dict params. Inputs are as described in the CCIT
        package.
        '''
        self.params = locals()
        del self.params['self']
    def run_test(self,x, y, z):
        '''
        compute the test statistics and pvalues

        :param x: input data for x
        :type x: ndarray
        :param y: input data for y
        :type y: ndarray
        :param z: input data for z
        :type z: ndarray

        :return: Returns a tuple of the form (None,pvalue) (instead of simply pvalue to maintain consistency with other
        tests in the package)
        :rtype: tuple of the form (None,float)
        '''
        return (None, CCIT.CCIT(np.expand_dims(x,axis=1), np.expand_dims(y,axis=1), z, **self.params))