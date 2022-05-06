import numpy as np
import itertools
from models.rosenbrock_analytic_proper_sample import rosenbrock_analytic
from source.stein_experiments import SVI
import os
import copy
import scipy
import numdifftools as nd

# t = timeit.Timer(functools.partial(self.stein.getGNHessianMinusLogPosterior_ensemble_new, self.stein.particles))
# t_contract = timeit.Timer(functools.partial(self.H_bar_contract, GN_Hmlpt, kernel, gradKernel))
class Test_sanity:
    def setup(self):
        model = rosenbrock_analytic()
        self.rtol = 1e-8
        self.atol = 1e-8
        self.optimizeMethod = 'SVN'
        # self.nParticles = 4
        self.nParticles = 5
        self.nIterations = 1
        self.stein = SVI(model = model, nParticles=self.nParticles, nIterations=self.nIterations, optimizeMethod = self.optimizeMethod)
        self.stein.SVGD_stochastic_correction = False
        self.X = np.copy(self.stein.model.newDrawFromPrior(self.nParticles))

    def test_gradient(self):
        for n in range(self.nParticles):
            particle = self.X[n]
            test_a = self.stein.getGradientMinusLogPosterior_individual(particle)
            test_b = nd.Gradient(self.stein.getMinusLogPosterior_individual)(particle)
            assert(np.allclose(test_a, test_b, rtol=self.rtol, atol=self.atol))

    def test_Hessian(self):
        for n in range(self.nParticles):
            particle = self.X[n]
            test_a = self.stein.getHessianMinusLogPosterior_individual(particle)
            test_b = nd.Hessian(self.stein.getMinusLogPosterior_individual)(particle)
            assert(np.allclose(test_a, test_b, rtol=self.rtol, atol=self.atol))

    def test_gradient_GN_Hessian(self):
        for n in range(self.nParticles):
            particle = self.X[n]
            test_a = self.stein.model.getGradientGNHessianMinusLogLikelihood_individual(particle)
            tmp = (nd.Gradient(self.stein.getGNHessianMinusLogPosterior_individual)(particle))
            # It seems as though numdifftools puts the derivative in the first slot instead of the last as we wanted.
            # Thus a swap-axes is needed
            test_b = np.einsum('ijk -> kij', tmp) # proof of concept
            assert(np.allclose(test_a, test_b, rtol=self.rtol, atol=self.atol))

# For debugging purposes
def main():
    a = Test_sanity()
    a.setup()
    # a.form_gradK_first_slot()
    # a.test_divD()
    # a.test_K_action_mnbd()
    # a.test_getSVN_Direction()
    # a.test_numerical_gradient_h_ij_BD()
    # a.formGradH_BD_second_slot()
    # a.test_gradHBD_action_second_slot_mnbd()
    # a.test_gradient()
    a.test_gradient_GN_Hessian()
if __name__ is '__main__':
    main()