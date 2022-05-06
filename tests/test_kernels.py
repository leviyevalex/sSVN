import numpy as np
import numdifftools as nd
from tools.kernels import Gaussian_identity, IMQ_identity, Gaussian_metric
from tools.kernels import getGaussianKernelWithDerivatives_identity as clean_calculation
from tools.kernels import getIMQ_kernelWithDerivatives_identity, getGaussianKernelWithDerivatives_metric
from tools.kernels import getIMQ_metricWithDerivatives
from tools.kernels import IMQ_metric
from tools.kernels import linear, linear_metric, random
from tools.kernels import getLinear_metricWithDerivatives
import itertools
class Test_kernel():
    def setup(self):
        self.DoF = 2
        self.n = 5
        self.m = 6
        M=np.array([[5, 1], [1, 3]])
        # self.kernel = Gaussian_metric(h=2, M=np.array([[5, 1], [1, 3]]))
        # self.kernel = IMQ_metric(M=M, h=2)
        # self.kernel = linear()
        self.kernel = linear_metric(M)
        # self.kernel = random(h=1, l=5, D=self.DoF)
        # self.kernel = Gaussian_identity_metric()
        # self.kernel = IMQ()
        self.X = np.random.rand(self.m, self.DoF)
        self.Y = np.random.rand(self.n, self.DoF)

    def test_gkx1(self):
        gk_num = nd.Jacobian(self.kernel.k)
        for m, n in itertools.product(range(self.m), range(self.m)):
            assert np.allclose(gk_num(self.X[m], self.X[n]), self.kernel.gk1(self.X[m], self.X[n]))

    def test_gkx2(self):
        temp = lambda y, x: self.kernel.k(x,y)
        gk2_num = nd.Jacobian(temp)
        for m, n in itertools.product(range(self.m), range(self.m)):
            assert np.allclose(gk2_num(self.X[m], self.X[n]), self.kernel.gk2(self.X[m], self.X[n]))
    def test_hess_kernel(self):
        hessk_num = nd.Hessian(self.kernel.k)
        for m, n in itertools.product(range(self.m), range(self.m)):
            assert np.allclose(hessk_num(self.X[m], self.X[n]), self.kernel.hessk(self.X[m], self.X[n]))
    def test_kernel_ensemble(self):
        kx = self.kernel.kx(self.X, self.Y)
        for m, n in itertools.product(range(self.m), range(self.n)):
            assert np.allclose(kx[m, n], self.kernel.k(self.X[m], self.Y[n]))
    def test_grad_kernel_ensemble(self):
        gkx = self.kernel.gkx(self.X, self.Y)
        for m, n in itertools.product(range(self.m), range(self.n)):
            assert np.allclose(gkx[m,n], self.kernel.gk(self.X[m], self.Y[n]))
    def test_hessian_kernel_ensemble(self):
         hesskx = self.kernel.hesskx(self.X, self.Y)
         for m, n in itertools.product(range(self.m), range(self.n)):
             assert np.allclose(hesskx[m,n], self.kernel.hessk(self.X[m], self.Y[n]))

    def test_getGaussianKernelWithDerivatives_identity(self):
        kernel = Gaussian_identity()
        h = kernel.h
        kx1, gkx1, hesskx1 = clean_calculation(self.X, h, get_hesskx=True) # This line is all that needs to be changed.
        kx2 = kernel.kx(self.X, self.X)
        gkx2 = kernel.gkx(self.X, self.X)
        hesskx2 = kernel.hesskx(self.X, self.X)
        assert np.allclose(kx1, kx2)
        assert np.allclose(gkx1, gkx2)
        assert np.allclose(hesskx1, hesskx2)

    def test_getIMQ_KernelWithDerivatives(self):
        kernel = IMQ_identity()
        beta = kernel.beta
        c = kernel.c
        kx1, gkx1, hesskx1 = getIMQ_kernelWithDerivatives_identity(self.X, c=c, beta=beta, get_hesskx=True) # This line is all that needs to be changed.
        kx2 = kernel.kx(self.X, self.X)
        gkx2 = kernel.gkx(self.X, self.X)
        hesskx2 = kernel.hesskx(self.X, self.X)
        assert np.allclose(kx1, kx2)
        assert np.allclose(gkx1, gkx2)
        assert np.allclose(hesskx1, hesskx2)

    def test_getGaussianMetricKernelWithDerivatives(self):
        kernel = Gaussian_metric(h=2, M=np.array([[5, 1], [1, 3]]))
        h = kernel.h
        M = kernel.M
        kx1, gkx1, hesskx1 = getGaussianKernelWithDerivatives_metric(self.X, h=h, M=M, get_hesskx=True) # This line is all that needs to be changed.
        kx2 = kernel.kx(self.X, self.X)
        gkx2 = kernel.gkx(self.X, self.X)
        hesskx2 = kernel.hesskx(self.X, self.X)
        assert np.allclose(kx1, kx2)
        assert np.allclose(gkx1, gkx2)
        assert np.allclose(hesskx1, hesskx2)
        tmp1, tmp2, tr_hesskx = getGaussianKernelWithDerivatives_metric(self.X, h=h, M=M, get_tr_hesskx=True)
        assert np.allclose(tr_hesskx, np.einsum('mnii->mn', hesskx2))
    def test_getIMQ_metricWithDerivatives(self):
        kernel = IMQ_metric(h=2, M=np.array([[5, 1], [1, 3]]))
        h = kernel.h
        M = kernel.M
        beta = -0.5
        c=1
        kx1, gkx1, hesskx1 = getIMQ_metricWithDerivatives(self.X, h=h, M=M, beta=beta, c=c, get_hesskx=True) # This line is all that needs to be changed.
        kx2 = kernel.kx(self.X, self.X)
        gkx2 = kernel.gkx(self.X, self.X)
        hesskx2 = kernel.hesskx(self.X, self.X)
        assert np.allclose(kx1, kx2)
        assert np.allclose(gkx1, gkx2)
        assert np.allclose(hesskx1, hesskx2)
        tmp1, tmp2, tr_hesskx = getIMQ_metricWithDerivatives(self.X, h=h, M=M, beta=beta, c=c, get_tr_hesskx=True)
        assert np.allclose(tr_hesskx, np.einsum('mnii->mn', hesskx2))

    def test_getLinear_metricWithDerivatives(self):
        M=np.array([[5, 1], [1, 3]])
        kernel = linear_metric(M)
        kx, gkx1, gkx2, tr_hesskx11, hesskx12, hesskx11 = getLinear_metricWithDerivatives(self.X, M=M,
                                                          get_tr_hesskx11=True,
                                                          get_hesskx12=True,
                                                          get_hesskx11=True)
        kx_test = kernel.kx(self.X, self.X)
        gkx_test = kernel.gkx(self.X, self.X)
        hesskx_test = kernel.hesskx(self.X, self.X)
        assert np.allclose(kx, kx_test)
        assert np.allclose(gkx1, gkx_test)
        assert np.allclose(hesskx1, hesskx2)
if __name__ is '__main__':
    a = Test_kernel()
    a.setup()
    a.test_kernel_ensemble()
    a.test_grad_kernel()
    a.test_grad_kernel_ensemble()
def main():
    from pathlib import Path