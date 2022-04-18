import numpy as np
import pytest
import itertools
from models.gauss_analytic_1D import gauss_analytic_1D
from source.stein_experiments import SVI
import os


class Test_sanity:
    def setup(self):
        # Chose to redeclare. Can take any method and stick back into main methods easily if need be.
        output_dir = os.path.dirname(os.path.abspath(__file__)) + '/outdir'
        model = gauss_analytic_1D()
        self.rtol = 1e-15
        self.atol = 1e-15
        self.optimizeMethod = 'SVN'
        self.nParticles = 2
        self.nIterations = 1
        self.stein = SVI(model = model, nParticles=self.nParticles, nIterations=self.nIterations, optimizeMethod = self.optimizeMethod, output_dir=output_dir)
        self.particles = np.copy(self.stein.particles) # copy because particles are updated after apply()
        self.DoF = self.stein.DoF
        self.stein.apply()
        self.metric_ensemble = self.stein.metric_ensemble
        self.gradient_k_gram = self.stein.gradient_k_gram
        self.GN_Hmlpt = self.stein.GN_Hmlpt
        self.gmlpt = self.stein.gmlpt
        self.M = self.stein.M
        self.H = self.stein.H
        self.k_gram = self.stein.k_gram
        self.bandwidth = self.stein.bandwidth
        self.alphas = self.stein.alphas
        self.particle0 = self.particles[:, 0]
        self.particle1 = self.particles[:, 1]

    def test_likelihood(self):
        """
        Check if the likelihood matches analytic predictions
        """
        f = lambda x: 0.5 * x ** 2
        assert(np.isclose(f(self.particle0), self.stein.model.getMinusLogLikelihood_individual(self.particle0)))
        assert(np.isclose(f(self.particle1), self.stein.model.getMinusLogLikelihood_individual(self.particle1)))

    def test_posterior(self):
        f = lambda x: 0.5 * x ** 2 + 1 # flat prior adds the 1
        assert(np.isclose(f(self.particle0), self.stein.getMinusLogPosterior_individual(self.particle0)))
        assert(np.isclose(f(self.particle1), self.stein.getMinusLogPosterior_individual(self.particle1)))

    def test_grad_likelihood(self):
        assert(np.isclose(self.particle0, self.stein.model.getGradientMinusLogLikelihood_individual(self.particle0)))
        assert(np.isclose(self.particle1, self.stein.model.getGradientMinusLogLikelihood_individual(self.particle1)))

    def test_grad_posterior(self):
        grad_posterior0 = self.stein.getGradientMinusLogPosterior_individual(self.particle0)
        grad_posterior1 = self.stein.getGradientMinusLogPosterior_individual(self.particle1)
        assert(np.isclose(self.particle0, grad_posterior0))
        assert(np.isclose(self.particle1, grad_posterior1))
        grad_posterior_ensemble = self.stein.getGradientMinusLogPosterior_ensemble(self.particles)
        assert(np.isclose(self.particle0, grad_posterior_ensemble[:, 0]))
        assert(np.isclose(self.particle1, grad_posterior_ensemble[:, 1]))

    def test_hessian(self):
        assert(np.isclose(self.stein.getGNHessianMinusLogPosterior_individual(self.particle0), 1))
        assert(np.isclose(self.stein.getGNHessianMinusLogPosterior_individual(self.particle1), 1))
        assert(np.allclose(self.stein.getGNHessianMinusLogPosterior_ensemble(self.particles), 1))

    def test_M(self):
        assert(np.isclose(self.M, 1))

    def test_metric_ensemble(self):
        distance_squared_0_1 = (self.particle0 - self.particle1) ** 2
        assert(np.isclose(self.metric_ensemble[0, 1], distance_squared_0_1))
        assert(np.isclose(self.metric_ensemble[1, 0], distance_squared_0_1))

    def test_median_bandwidth(self):
        calculated = (np.abs(self.particle0 - self.particle1) ** 2) / np.log(self.nParticles)
        assert(np.isclose(calculated, self.bandwidth))

    def test_kernel(self):
        k = lambda x, y: np.exp(-1/self.bandwidth * (x - y) ** 2)
        assert(np.isclose(k(self.particle0, self.particle1), self.k_gram[0, 1]))
        assert(np.isclose(k(self.particle1, self.particle1), self.k_gram[1, 1]))

    def test_grad_kernel_is_repulsive(self):
        # x is to the left of y
        if self.particle0 < self.particle1:
            x = 0
            y = 1
        else:
            x = 1
            y = 0
        # Read as follows: force on 2 by 1
        assert(self.gradient_k_gram[x, y] > 0)
        assert(self.gradient_k_gram[y, x] < 0)

    def test_grad_kernel(self):
        # Remember, these functions are with respect to vectors in \chi, not integers!
        gradk = lambda x, y: np.exp(-1/self.bandwidth * (x - y) ** 2) * (x-y) * (-2 / self.bandwidth)
        # kgramgrad = lambda m, n : -2 * self.k_gram[m, n] * (self.particles[:, m] - self.particles[:, n]) / self.bandwidth
        assert(np.isclose(gradk(self.particle0, self.particle1), self.gradient_k_gram[0, 1, :]))
        assert(np.isclose(gradk(self.particle1, self.particle0), self.gradient_k_gram[1, 0, :]))



# For debugging purposes
def main():
    a = Test_sanity()
    a.setup()
    a.test_grad_kernel()

if __name__ is '__main__':
    main()
