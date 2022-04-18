import numpy as np
import numdifftools as nd
# from tools.kernels import get_linear_metric as get_kernel, class_linear_metric as class_kernel
# from tools.kernels import get_IMQ_metric as get_kernel, class_IMQ_metric as class_kernel
from tools.kernels import get_randomRBF_metric as get_kernel, class_random_feature_RBF_metric as class_kernel
# from tools.kernels import get_pointwise_preconditioned_RBF_v2 as get_kernel, class_pointwise_preconditioned_RBF as class_kernel
import itertools
class Test_SVI_kernels():
    def setup(self):
        self.DoF = 2
        self.m = 6 # Number of particles
        self.M = np.array([[5, 1], [1, 3]])
        self.Ml = np.random.uniform(1, 2, (self.m, self.DoF, self.DoF)) * np.eye(self.DoF)[np.newaxis, ...]
        self.X = np.random.rand(self.m, self.DoF)
        # self.M = None
        # self.kernel = class_kernel(M=self.M).k

        # RBF random features
        self.l = 5
        self.h = 2
        self.w = np.random.multivariate_normal(np.zeros(self.DoF), np.eye(self.DoF), self.l)
        self.v = np.random.uniform(0, 2 * np.pi)
        self.kernel = class_kernel(M=self.M, D=self.DoF).k # For the random RBF
        # self.kernel = class_kernel().k
        # self.kernel = class_kernel(M=self.M).k # Linear
        # PPRBF Kernel
        # self.kernel = class_kernel(X=self.X, Ml=self.Ml).k

        # self.kx, self.gkx1 = get_kernel(X=self.X, Ml=self.Ml, get_kx=True, get_gkx1=True)
        #
        self.kx, self.gkx1, self.gkx2, self.tr_hesskx11, self.hesskx11, self.hesskx12 = \
            get_kernel(X=self.X,
                       M=self.M,
                       get_kx=True,
                       get_gkx1=True,
                       get_gkx2=True,
                       get_tr_hesskx11=True,
                       get_hesskx11=True,
                       get_hesskx12=True)

    def test_kx(self):
        for m, n in itertools.product(range(self.m), range(self.m)):
            assert np.allclose(self.kx[m, n], self.kernel(self.X[m], self.X[n]))

    def test_gkx1(self):
        gkx1_num = nd.Jacobian(self.kernel)
        for m, n in itertools.product(range(self.m), range(self.m)):
            assert np.allclose(gkx1_num(self.X[m], self.X[n]), self.gkx1[m, n])

    def test_gkx2(self):
        temp = lambda y, x: self.kernel(x, y)
        gkx2_num = lambda x, y: nd.Jacobian(temp)(y, x)
        for m, n in itertools.product(range(self.m), range(self.m)):
            assert np.allclose(gkx2_num(self.X[m], self.X[n]), self.gkx2[m, n])

    def test_tr_hesskx11(self):
        hesskx11_num = nd.Hessian(self.kernel)
        for m, n in itertools.product(range(self.m), range(self.m)):
            assert np.allclose(np.trace(hesskx11_num(self.X[m], self.X[n])), self.tr_hesskx11[m, n])

    def test_hesskx11(self):
        hesskx11_num = nd.Hessian(self.kernel)
        for m, n in itertools.product(range(self.m), range(self.m)):
            assert np.allclose(hesskx11_num(self.X[m], self.X[n]), self.hesskx11[m, n])

    def test_hesskx12(self):
        tmp1 = nd.Jacobian(lambda y, x: self.kernel(x, y))
        hesskx12_num = nd.Jacobian(lambda x,y: tmp1(y, x))
        for m, n in itertools.product(range(self.m), range(self.m)):
            assert np.allclose(hesskx12_num(self.X[m], self.X[n]), self.hesskx12[m, n])

if __name__ is '__main__':
    a = Test_SVI_kernels()
    a.setup()
    a.test_kx()
    a.test_gkx1()
    a.test_gkx2()
    a.test_hesskx11()
    # a.test_kernel_ensemble()
    # a.test_grad_kernel()
    # a.test_grad_kernel_ensemble()
def main():
    from pathlib import Path