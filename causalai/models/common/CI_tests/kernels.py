from numpy import exp, median, sqrt, ndarray, eye
from typing import Tuple, List, Union, Optional
from numpy.random import permutation
from scipy.spatial.distance import cdist, pdist, squareform
from abc import abstractmethod
from numpy.linalg import pinv
import warnings

def YeqXifNone(func):
    def wrapper(s,X,Y=None):
        if Y is None:
            Y = X
        return func(s,X,Y)
    return wrapper

class KernelBase(object):
    def __init__(self, **kargs):
        self.__dict__.update(kargs)

    @abstractmethod
    def kernel(self, X: ndarray, Y: Optional[ndarray] = None) -> ndarray:
        """
        Returns the nxn kernel matrix
        """
        raise NotImplementedError()

    @staticmethod
    def centered_kernel(K: ndarray) -> ndarray:
        """
        Remove data mean by returning HKH, where H=I-1/n
        In the linear setting where K=XX', simple linear algebra shows that HKH is essentially the 
        kernel matrix (X-mu)(X-mu)' after centering the data matrix X, where each row is a sample.
        When using a non-linear kernel K, HKH centers the data in the kernel representation space.
        """
        n = K.shape[0]
        H = eye(n) - 1.0 / n
        return H.dot(K.dot(H))

    def kernel_matrix_regression(K: ndarray, Kz: ndarray, epsilon: float = 1e-5) -> ndarray:
        """
        Closed form Kernel Matrix Regression for computing the regression coefficient A.K.A that predicts K using Kz.
        Here A = Kz^-1, we use epsilon to avoid degenerate cases.
        See slide 14 of https://members.cbio.mines-paristech.fr/~jvert/talks/070529asmda/asmda.pdf for explaination.
        """
        A = epsilon * pinv(Kz + epsilon * eye(K.shape[0]))
        return A.dot(K.dot(A))

class LinearKernel(KernelBase):
    def __init__(self):
        KernelBase.__init__(self)
        
    @YeqXifNone
    def kernel(self, X: ndarray, Y: Optional[ndarray] = None) -> ndarray:
        """
        The linear kernel matrix: K(X,Y)=X'Y
        If Y==None, X=Y
        """
        return self.centered_kernel(X.dot(Y.T))

class GaussianKernel(KernelBase):
    def __init__(self, width='empirical'):
        KernelBase.__init__(self)
        assert width in ['empirical', 'median'], f'width must be one of ["empirical", "median"], but found {width}.'
        self.width = width

    @YeqXifNone
    def kernel(self, X: ndarray, Y: Optional[ndarray] = None) -> ndarray:
        """
        Gaussian kernel K(Xi,Yj) = exp(-0.5 * ||Xi-Yj||**2 / sigma**2)
        """
        if self.width == 'empirical':
            self.set_width_empirical(X)
        elif self.width == 'median':
            self.set_width_median(X)
        X = X.reshape(X.shape[0], 1, -1)
        Y = Y.reshape(1, Y.shape[0], -1)
        K = exp(-0.5 * self.width * ((X-Y)**2).sum(-1))
        return self.centered_kernel(K)

    def set_width_median(self, X: ndarray) -> None:
        if X.shape[0] > 1000: # use at max 1000 samples to find the median
            X = X[permutation(X.shape[0])[:1000], :]
        dists = squareform(pdist(X, 'euclidean'))
        median_dist = median(dists[dists > 0])
        self.width = 0.5 / (median_dist ** 2)
    def set_width_empirical(self, X: ndarray) -> None:
        if X.shape[0] < 200:
            width = 0.8
        elif X.shape[0] < 1200:
            width = 0.5
        else:
            width = 0.3
        theta = 1.0 / (width ** 2)
        self.width = theta * X.shape[1]
