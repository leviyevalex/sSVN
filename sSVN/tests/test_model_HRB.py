# import numpy as np
import autograd.numpy as np
import itertools
from models.HRD import hybrid_rosenbrock
from source.stein_experiments import SVI
import os
import copy
import scipy
import numdifftools as nd
from autograd import jacobian, hessian

# t = timeit.Timer(functools.partial(self.stein.getGNHessianMinusLogPosterior_ensemble_new, self.stein.particles))
# t_contract = timeit.Timer(functools.partial(self.H_bar_contract, GN_Hmlpt, kernel, gradKernel))
class Test_sanity:
    def setup(self):
        # self.n1 = 6
        # self.n2 = 3

        # self.n1 = 3
        # self.n2 = 2

        self.n1 = 2
        self.n2 = 1
        self.mu_rosen = 1
        self.a = 1 / 20
        self.b = np.ones((self.n2, self.n1-1)) * (100 / 20)
        self.model = hybrid_rosenbrock(self.n2, self.n1, self.mu_rosen, self.a, self.b)

        self.rtol = 1e-8
        self.atol = 1e-8
        self.optimizeMethod = 'SVN'
        self.nParticles = 5
        self.nIterations = 1

        self.stein = SVI(model = self.model, nParticles=self.nParticles, nIterations=self.nIterations, optimizeMethod = self.optimizeMethod)
        self.X = np.copy(self.stein.model.newDrawFromPrior(self.nParticles))
        # self.x_in = np.zeros(self.model.DoF) + 0.01

        # input vector manipulations and residual calculations
        self.x_in = np.random.rand(self.model.DoF) # Used for all tests that require a vector
        self.x_graph = np.insert(self.x_in[1:].reshape(self.n2, self.n1-1), 0, self.x_in[0], axis=1)
        self.rji = np.sqrt(2 * self.b) * (self.x_graph[:, 1:] - self.x_graph[:, :-1] ** 2)
        # mat = self.model.formKroneckerMatricies()
        mat = self.model.formKroneckerDeltas()
        self.Hrji = -1 * np.einsum('ji, jif, jie -> jief', np.sqrt(8 * self.b), mat['delta2'], mat['delta2'])
        tmp1 = np.einsum('ji, jie -> jie', np.sqrt(2 * self.b), mat['delta1'])
        tmp2 = -1 * np.einsum('ji, ji, jie -> jie', np.sqrt(2 * self.b), 2 * self.x_graph[:, :-1], mat['delta2'])
        self.grji = tmp1 + tmp2
        # self.x_in = np.random.normal(self.model.DoF) # Used for all tests that require a vector

    def test_mu(self):
        # Form index matrix using index function
        test_a = np.zeros((self.n2, self.n1))
        for j, i in itertools.product(range(self.n2), range(self.n1)):
            test_a[j,i] = self.model.mu(j,i)
        # Form index matrix manually
        tmp1 = np.arange(1, self.model.DoF).reshape(self.n2, self.n1-1)
        test_b = np.insert(tmp1, 0, 0, axis=1)
        assert np.allclose(test_a, test_b, rtol=self.rtol, atol=self.atol)

    def test_target_vectorization(self):
        test_a = self.model.getMinusLogLikelihood_individual(self.x_in)
        test_b = self.a * (self.x_in[0] - self.model.mu_rosen) ** 2
        for m, n in itertools.product(range(self.n2), range(self.n1 -1)):
            print(m, n)
            test_b += self.b[m,n] * (self.x_in[self.model.mu(m, n+1)] - self.x_in[self.model.mu(m, n)] ** 2) ** 2
        assert np.allclose(test_a, test_b, rtol=self.rtol, atol=self.atol)

    def test_2D_case(self):
        x = np.random.rand(2)
        n1 = 2
        n2 = 1
        mu_rosen = 1
        a = 1 / 20
        b = np.ones((n2, n1-1)) * (100 / 20)
        HRD_2D = lambda x: a * (x[0] - mu_rosen) ** 2 + b * (x[1] - x[0] ** 2) ** 2
        test_a = HRD_2D(x)
        model = hybrid_rosenbrock(n2, n1, mu_rosen, a, b)
        test_b = model.getMinusLogLikelihood_individual(x)
        assert np.allclose(test_a, test_b, rtol=self.rtol, atol=self.atol)

    def test_5D_case(self):
        x = np.random.rand(5)
        n1 = 3
        n2 = 2
        mu_rosen = 1
        a = 1 / 20
        b = np.ones((n2, n1-1)) * (100 / 20)
        HRD_5D = lambda x: a * (x[0] - mu_rosen) ** 2 + b[0,0] * (x[1] - x[0] ** 2) ** 2 \
                                                        + b[0,1] * (x[2] - x[1] ** 2) ** 2 \
                                                        + b[1,0] * (x[3] - x[0] ** 2) ** 2 \
                                                        + b[1,1] * (x[4] - x[3] ** 2) ** 2
        test_a = HRD_5D(x)
        model = hybrid_rosenbrock(n2, n1, mu_rosen, a, b)
        test_b = model.getMinusLogLikelihood_individual(x)
        assert np.allclose(test_a, test_b, rtol=self.rtol, atol=self.atol)

    def test_gradient(self):
        test_a = self.model.getGradientMinusLogLikelihood_individual(self.x_in)
        test_b = jacobian(self.model.getMinusLogLikelihood_individual)(self.x_in)
        assert np.allclose(test_a, test_b, rtol=self.rtol, atol=self.atol)

    def test_Hessian(self):
        test_a = self.model.getHessianMinusLogLikelihood_individual(self.x_in)
        test_b = hessian(self.model.getMinusLogLikelihood_individual)(self.x_in)
        assert np.allclose(test_a, test_b, rtol=self.rtol, atol=self.atol)

    def test_GN_Hessian(self):
        test_a = self.model.getGNHessianMinusLogLikelihood_individual(self.x_in)
        test_b = hessian(self.model.getMinusLogLikelihood_individual)(self.x_in) - np.einsum('ji, jief -> ef', self.rji, self.Hrji)
        assert np.allclose(test_a, test_b, rtol=self.rtol, atol=self.atol)

    def test_gradient_GN_Hessian(self):
        test_a = self.model.getGradientGNHessianMinusLogLikelihood_individual(self.x_in)
        test_b = np.einsum('ijk -> kij', jacobian(hessian(self.model.getMinusLogLikelihood_individual))(self.x_in)) - np.einsum('jig, jief -> efg', self.grji, self.Hrji)
        assert np.allclose(test_a, test_b, rtol=self.rtol, atol=self.atol)


# test vectorization
# test_b = self.a * (self.x_in[0] - self.model.mu_rosen) ** 2
# for j, I in itertools.product(range(self.n2), np.arange(1, self.n1)):
#     print(j, I)
#     test_b += self.b[j,I-1] * (self.x_in[self.model.mu(j, I)] - self.x_in[self.model.mu(j, I - 1)] ** 2) ** 2
# assert np.allclose(test_a, test_b, rtol=self.rtol, atol=self.atol)

# # Alternative form
# test_c = self.a * (self.x_in[0] - self.model.mu_rosen) ** 2
# for m, n in itertools.product(range(self.n2), range(self.n1 -1)):
#     print(m, n)
#     test_c += self.b[m,n-1] * (self.x_in[self.model.mu(m, n+1)] - self.x_in[self.model.mu(m, n)] ** 2) ** 2
# assert np.allclose(test_a, test_c, rtol=self.rtol, atol=self.atol)

# For debugging purposes
def main():
    a = Test_sanity()
    a.setup()
    a.test_mu()
    a.test_target_vectorization()
    a.test_2D_case()
    a.test_5D_case()
    a.test_gradient()
    a.test_Hessian()
    a.test_GN_Hessian()
    a.test_gradient_GN_Hessian()

if __name__ is '__main__':
    main()