import numpy as np
from numpy import sqrt, ndarray
from typing import Tuple, List, Union, Optional
from numpy.linalg import eigh, eigvalsh
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel
from typing import Type

from .kernels import KernelBase, LinearKernel, GaussianKernel


class KCI:
    """
    Kernel-based Conditional Independence (KCI) test.
    Original implementation: http://people.tuebingen.mpg.de/kzhang/KCI-test.zip

    References
    ----------
    [1] K. Zhang, J. Peters, D. Janzing, and B. Schölkopf, "A kernel-based conditional independence test and application in causal discovery," In UAI 2011.
    """

    def __init__(self, Xkernel: KernelBase=GaussianKernel(),
                       Ykernel: KernelBase=GaussianKernel(),
                       Zkernel: KernelBase=GaussianKernel(),
                       null_space_size: int=5000,
                       approx: bool=True,
                       chunk_size: int = 1000):
        """
        KCI test constructor.

        :param Xkernel: kernel class instance for input data x. Available options are GaussianKernel and LinearKernel.
        :type Xkernel: KernelBase object
        :param Ykernel: kernel class instance for input data y. Available options are GaussianKernel and LinearKernel.
        :type Ykernel: KernelBase object
        :param Zkernel: kernel class instance for input data z (conditional variables). Available options are GaussianKernel and LinearKernel.
        :type Zkernel: KernelBase object
        :param null_space_size: sample size in simulating the null distribution (default=5000).
        :type null_space_size: int
        :param approx: whether to use gamma approximation (default=True).
        :type approx: bool
        :param chunk_size: if number of data samples is more than chunk_size (default=1000), only extract the block-wise diagonal kernel matrix
            of the full kernel matrix to save memory and computation.
        :type chunk_size: int
        """
        self.Xkernel = Xkernel
        self.Ykernel = Ykernel
        self.Zkernel = Zkernel
        self.null_space_size = null_space_size
        self.epsilon_x = 1e-5 # Too large values lead to false causal link detection
        self.epsilon_y = 1e-5 
        self.thres = 1e-5
        self.approx = approx
        self.chunk_size = chunk_size

    def run_test(self, data_x: ndarray=None, data_y: ndarray=None, data_z: Optional[ndarray]=None) -> Tuple[float, float]:
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
        if data_z is not None:
            assert data_x.shape[0]==data_y.shape[0]==data_z.shape[0],\
                 f'The number of data points (index 0) of data_x, data_y, data_z must match. Found {data_x.shape[0]}, {data_y.shape[0]}, {data_z.shape[0]} respectively.'
        else:
            assert data_x.shape[0]==data_y.shape[0],\
                 f'The number of data points (index 0) of data_x and data_y must match. Found {data_x.shape[0]} and {data_y.shape[0]} respectively.'
        if len(data_x.shape)==1:
            data_x = data_x.reshape(-1,1)
        if len(data_y.shape)==1:
            data_y = data_y.reshape(-1,1)

        # To handle large scale data, the kernel regression model is applied to the block diagonal chunks of the full kernel matrix
        test_stat_all, pvalue_all = [],[]
        for chunk_id in range(1 + data_x.shape[0]//self.chunk_size):
            if chunk_id*self.chunk_size<data_x.shape[0]-1:
                data_x_i = data_x[chunk_id*self.chunk_size: (chunk_id+1)*self.chunk_size]
                data_y_i = data_y[chunk_id*self.chunk_size: (chunk_id+1)*self.chunk_size]
                data_z_i = data_z[chunk_id*self.chunk_size: (chunk_id+1)*self.chunk_size] if data_z is not None else None

                Kx_i = self.Xkernel.kernel(data_x_i)
                Ky_i = self.Ykernel.kernel(data_y_i)
                Kz_i = self.Zkernel.kernel(data_z_i) if data_z is not None else None

                test_stat, KxR, KyR = self._V_statistic(Kx_i, Ky_i, Kz_i)
                if data_z is not None: 
                    xy_pc_dot_prod, size_u = self._principle_component_dot_prod(KxR, KyR)

                    if self.approx:
                        kappa, theta = self._get_kappa(xy_pc_dot_prod)
                        pvalue = 1 - stats.gamma.cdf(test_stat, kappa, 0, theta)
                    else:
                        null_samples = self._null_sample_spectral(xy_pc_dot_prod, size_u, Kx.shape[0])
                        pvalue = sum(null_samples > test_stat) / float(self.null_space_size)
                else: # unconditional CI test
                    if self.approx:
                        kappa, theta = self._get_kappa_uncond(Kx_i, Ky_i)
                        pvalue = 1 - stats.gamma.cdf(test_stat, kappa, 0, theta)
                    else:
                        null_samples = self._null_sample_spectral_uncond(Kx_i, Ky_i)
                        pvalue = sum(null_samples > test_stat) / float(self.null_space_size)


                test_stat_all.append(test_stat)
                pvalue_all.append(pvalue)
        return np.mean(np.array(test_stat_all)), np.mean(np.array(pvalue_all))

    def _V_statistic(self, Kx: ndarray, Ky: ndarray, Kz: Optional[ndarray]) -> Tuple[float,ndarray,ndarray]:
        """
        Compute V test statistic from kernel matrices Kx and Ky

        :param Kx: kernel matrix for data_x 
        :param Kx: ndarray
        :param Ky: kernel matrix for data_y
        :param Ky: ndarray
        :param Kz: kernel matrix for data_z
        :param Kz: ndarray

        :return: A tuple of length 3:

            - Vstat: V statistics (float)
            
            - KxR: kernel regression matrix for data_x if Kz is not None else Kx (ndarray)
            
            - KyR: kernel regression matrix for data_y if Kz is not None else Ky (ndarray)
        :rtype: tuple
        """
        if Kz is not None:
            KxR = KernelBase.kernel_matrix_regression(Kx, Kz, self.epsilon_x)
            KyR = KernelBase.kernel_matrix_regression(Ky, Kz, self.epsilon_y)
            Vstat = np.sum(KxR * KyR)
            return Vstat, KxR, KyR
        Vstat = np.sum(Kx * Ky)
        return Vstat, Kx, Ky

    def _principle_component_dot_prod(self, Kx: ndarray, Ky: ndarray) -> Tuple[ndarray, int]:
        """
        Compute eigenvalues for null distribution estimation

        :param Kx: ndarray
            kernel matrix for data_x
        :param Ky: ndarray 
            kernel matrix for data_y

        :return: tuple
            xy_pc_dot_prod: ndarray, product of the eigenvectors of Kx and Ky
            size_u: int, number of eigenvector products

        """
        wx, vx = eigh(0.5 * (Kx + Kx.T))
        wy, vy = eigh(0.5 * (Ky + Ky.T))
        idx = np.argsort(-wx)
        idy = np.argsort(-wy)
        wx = wx[idx]
        vx = vx[:, idx]
        wy = wy[idy]
        vy = vy[:, idy]
        vx = vx[:, wx > np.max(wx) * self.thres]
        wx = wx[wx > np.max(wx) * self.thres]
        vy = vy[:, wy > np.max(wy) * self.thres]
        wy = wy[wy > np.max(wy) * self.thres]
        vx = vx.dot(np.diag(np.sqrt(wx)))
        vy = vy.dot(np.diag(np.sqrt(wy)))

        # calculate pair-wise product
        N = Kx.shape[0]
        num_eigx = vx.shape[1]
        num_eigy = vy.shape[1]
        size_u = num_eigx * num_eigy
        uu = np.zeros((N, size_u))
        for i in range(0, num_eigx):
            for j in range(0, num_eigy):
                uu[:, i * num_eigy + j] = vx[:, i] * vy[:, j]

        if size_u > N:
            xy_pc_dot_prod = uu.dot(uu.T)
        else:
            xy_pc_dot_prod = uu.T.dot(uu)

        return xy_pc_dot_prod, size_u

    def _null_sample_spectral(self, xy_pc_dot_prod: ndarray, size_u: int, N: int) -> ndarray:
        """
        Simulate data from null distribution

        :param xy_pc_dot_prod: ndarray
            product of the eigenvectors of Kx and Ky
        :param size_u: int 
            number of eigenvector products
        :param N: int
            sample size

        :return: ndarray
            samples: samples from the null distribution

        """
        eig_uu = eigvalsh(xy_pc_dot_prod)
        eig_uu = -np.sort(-eig_uu)
        eig_uu = eig_uu[0:np.min((N, size_u))]
        eig_uu = eig_uu[eig_uu > np.max(eig_uu) * self.thres]

        f_rand = np.random.chisquare(1, (eig_uu.shape[0], self.null_space_size))
        samples = eig_uu.T.dot(f_rand)
        return samples

    def _null_sample_spectral_uncond(self, Kx, Ky):
        """
        Simulate data from null distribution

        :param Kx: ndarray
            centered kernel matrix for data_x
        :param Ky: ndarray 
            centered kernel matrix for data_y

        :return: ndarray
            null_dstr: samples from the null distribution

        """
        s = Kx.shape[0]
        if s > 1000:
            num_eig = np.int(np.floor(s/2.))
        else:
            num_eig = s
        eigval_x = (-np.sort(-eigvalsh(Kx)))[0:num_eig]
        eigval_y = (-np.sort(-eigvalsh(Ky)))[0:num_eig]
        eigval_prod = np.dot(eigval_x.reshape(-1, 1), eigval_y.reshape(1, -1)).reshape((-1, 1))
        eigval_prod = eigval_prod[eigval_prod > eigval_prod.max()* self.thres]
        f_rand = np.random.chisquare(1, (eigval_prod.shape[0], self.null_space_size))
        samples = eigval_prod.T.dot(f_rand) / s
        return samples

    def _get_kappa(self, xy_pc_dot_prod: ndarray) -> Tuple[float,float]:
        """
        Get approximate parameters for the gamma distribution

        :param xy_pc_dot_prod: ndarray
            product of the eigenvectors of Kx and Ky

        :return: tuple
            kappa, theta: parameters of the gamma distribution

        """
        mean = np.trace(xy_pc_dot_prod)
        var = 2 * np.trace(xy_pc_dot_prod.dot(xy_pc_dot_prod))
        kappa = mean**2/var
        theta = var/mean
        return kappa, theta

    def _get_kappa_uncond(self, Kx, Ky):
        """
        Get approximate parameters for the gamma distribution

        :param Kx: ndarray
            kernel matrix for data_x
            Ky: kernel matrix for data_y

        :return: tuple
            kappa, theta: parameters of the gamma distribution

        """
        s = Kx.shape[0]
        mean = np.trace(Kx)* np.trace(Ky)/s
        var = 2*np.sum(Kx**2)* np.sum(Ky**2)/(s**2)
        kappa = mean**2/var
        theta = var/mean
        return kappa, theta
